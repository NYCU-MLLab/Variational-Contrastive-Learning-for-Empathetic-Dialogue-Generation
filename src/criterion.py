import math
import numpy as np
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.utils import move_to_cuda


@register_criterion('ved_loss')
class VEDLoss(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        # ngram loss related
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss
        # kl-loss related
        self.target_kl = args.target_kl
        self.kl_loss_weight = args.kl_loss_weight
        self.disable_kl_loss = args.disable_kl_loss
        # bag-of-word loss and masked-lm loss
        self.cls_bow_loss_weight = args.cls_bow_loss_weight
        self.latent_bow_loss_weight = args.latent_bow_loss_weight
        self.masked_lm_loss_weight = args.masked_lm_loss_weight
        # whether to use tfidf weights in bag-of-loss module
        self.use_tfidf_weights = args.use_tfidf_weights
        if self.use_tfidf_weights:
            assert args.tfidf_model_path is not None and args.tfidf_dictionary_path is not None
            self.tfidf_model_path = args.tfidf_model_path
            self.tfidf_dictionary_path = args.tfidf_dictionary_path
            self.tfidf_model = None
            self.tfidf_dictionary = None
            self.init_tfidf_model()
        # whether class losses added (Empathetic new)
        self.emotion_labels = args.emotion_labels
        self.action_labels = args.action_labels
        self.add_cl_loss = args.add_cl_loss
        self.cl_coherence = args.cl_coherence
        self.cl_emotion = args.cl_emotion
        self.cl_action = args.cl_action
        self.temp_scale_class = args.temp_scale_class
        self.temp_scale_instance = args.temp_scale_instance
        self.cl_style = args.cl_style
        self.nll_weight = args.nll_weight
        self.cl_weight = args.cl_weight
        self.class_weight = args.class_weight
        self.cl_emotion_weight = args.cl_emotion_weight
        self.cl_action_weight = args.cl_action_weight
        self.cl_coherence_weight = args.cl_coherence_weight
        
        self.num_update= 0
        self.update_freq = args.update_freq
        
    @staticmethod
    def add_args(parser):
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true', default=False,
                            help='only compute basic stat')
        parser.add_argument('--use-tfidf-weights', action='store_true', default=False)
        parser.add_argument('--tfidf-model-path', type=str, help='tfidf model path')
        parser.add_argument('--tfidf-dictionary-path', type=str, help='tfidf dictionary path')
        
        # Empathetich new
        parser.add_argument('--add-cl-loss', default=False, type=bool, help='using contrastive loss or not') 
        parser.add_argument('--cl-coherence', default=False, type=bool, help='using contrastive with coherence or not') 
        parser.add_argument('--cl-emotion', default=False, type=bool, help='using contrastive with emotion or not') 
        parser.add_argument('--cl-action', default=False, type=bool, help='using contrastive with action or not') 
        parser.add_argument('--temp-scale-class', default=0.5, type=float, help='temperatre scale used in class realtion contrastive loss') 
        parser.add_argument('--temp-scale-instance', default=0.05, type=float, help='temperatre scale used in instance realtion contrastive loss') 
        parser.add_argument('--cl-style', default=False, type=bool, help='use contrastive loss as style transfer')
        parser.add_argument('--nll-weight', default=1, type=float, help='scaling nll loss')
        parser.add_argument('--cl-weight', default=0, type=float, help='scaling contrastive loss')
        parser.add_argument('--class-weight', default=1, type=float, help='scaling class prediction loss')
        parser.add_argument('--cl-emotion-weight', default=1, type=float, help='scaling contrastive emotion loss')
        parser.add_argument('--cl-action-weight', default=1, type=float, help='scaling contrastive action loss')
        parser.add_argument('--cl-coherence-weight', default=1, type=float, help='scaling contrastive coherence loss')
        
        
        
    def init_tfidf_model(self):
        from gensim.models import TfidfModel
        from gensim.corpora import Dictionary
        self.tfidf_model = TfidfModel.load(self.tfidf_model_path)
        self.tfidf_dictionary = Dictionary.load(self.tfidf_dictionary_path)
        print('| loading tfidf model from {} ...'.format(self.tfidf_model_path))

    def cal_tfidf_weights(self, targets, seq_len, eps=1e-4):
        # this function should input a torch tensor [batch, seq_len],
        # and output a tf-idf weight matrix torch tensor [batch, seq_len]
        # this function may effect the whole efficiency, since it can be pre-computed and cached ...
        _targets = targets.clone().cpu().numpy()
        tfidf_weights_map = [
            dict(tfidf_weight) for tfidf_weight in
            self.tfidf_model[[[(token_id, 1) for token_id in _target] for _target in _targets]]
        ]
        tfidf_weights = np.array(
            [[tfidf_weights_map[i].get(_item, eps) for _item in _target] for i, _target in enumerate(_targets)])
        # tfidf_weights = np.array([[weight[1] for weight in weights] for weights in self.tfidf_model[
        #     [[(token_id, 1) for token_id in _target] for _target in _targets]]])
        # use softmax will weaken the difference
        # tfidf_weights = F.softmax(tfidf_weights, dim=1)
        tfidf_weights = (tfidf_weights + eps) / (tfidf_weights.sum(axis=1)[:, None] + seq_len * eps) * seq_len
        tfidf_weights = move_to_cuda(torch.from_numpy(tfidf_weights.flatten('F')))
        return tfidf_weights

    def forward(self, model, sample, reduce=True):
        if self.cl_coherence or self.cl_action or self.cl_emotion or self.cl_style:
            self.num_update += (1 / self.update_freq[0])
        
        # execute forward method defined in model
        decoder_out, encoder_out = model(**sample['net_input'], return_all_hiddens=False)

        # n-gram predicting stream
        logits_list = decoder_out[0]
        
        # default targets fetch in sample directly
        targets = model.get_targets(sample, None)
        _, seq_len = targets.shape

        cls_bow_logits = encoder_out['cls_bow_logits']
        masked_logits = encoder_out['masked_logits']
        latent_bow_logits = encoder_out['latent_bow_logits']
        kl = encoder_out['kl']
        
        # Empathetic new
        emotion_logits = encoder_out['emotion_logits']  
        action_logits = encoder_out['action_logits']  
        embed_cl = encoder_out['embed_cl']
        embed_cl_com = encoder_out['embed_cl_com']
        embed_cl_action = encoder_out['embed_cl_action']
        embed_cl_action_com = encoder_out['embed_cl_action_com']
        embed_cl_emotion = encoder_out['embed_cl_emotion']
        embed_cl_emotion_com = encoder_out['embed_cl_emotion_com']
        embed_cl_style_anchor = encoder_out['embed_cl_style_anchor']
        embed_cl_style_pos = encoder_out['embed_cl_style_pos']
        embed_cl_style_com = encoder_out['embed_cl_style_com']
        

        # # find tok k most possible words in response
        # value, index = F.log_softmax(bow_logits, dim=1)[0].topk(25, largest=False, dim=1)
        # print(self.task.target_dictionary.string(index))
        # # see the target method's probability
        # print(bow_logits[0][targets[0]])

        # calculate bag of word loss
        cls_bow_loss = None
        if cls_bow_logits is not None:
            cls_bow_lprobs = F.log_softmax(cls_bow_logits, dim=-1, dtype=torch.float32)
            cls_bow_loss = F.nll_loss(
                input=cls_bow_lprobs.repeat(seq_len, 1),
                target=targets.transpose(1, 0).contiguous().view(-1),
                reduction='sum', ignore_index=self.padding_idx)

        if self.use_tfidf_weights:
            assert latent_bow_logits is not None, 'if `use_tfidf_weights`, latent_bow_logits should not be None!'

        latent_bow_loss = None
        if latent_bow_logits is not None:
            _, seq_len = targets.shape
            latent_bow_lprobs = F.log_softmax(latent_bow_logits, dim=-1, dtype=torch.float32)
            if self.use_tfidf_weights:
                latent_bow_loss = F.nll_loss(
                    input=latent_bow_lprobs.repeat(seq_len, 1),
                    target=targets.transpose(1, 0).contiguous().view(-1),
                    reduction='none', ignore_index=self.padding_idx)
                latent_bow_loss = torch.sum(torch.mul(latent_bow_loss, self.cal_tfidf_weights(targets, seq_len)))
            else:
                latent_bow_loss = F.nll_loss(
                    input=latent_bow_lprobs.repeat(seq_len, 1),
                    target=targets.transpose(1, 0).contiguous().view(-1),
                    reduction='sum', ignore_index=self.padding_idx)

        masked_lm_loss = None
        if masked_logits is not None:
            masked_targets = sample['masked_target'].long()
            masked_lprobs = F.log_softmax(masked_logits, dim=-1, dtype=torch.float32)
            masked_lm_loss = F.nll_loss(
                input=masked_lprobs.view(-1, masked_lprobs.size(-1)),
                target=masked_targets.view(-1),
                reduction='sum', ignore_index=self.padding_idx)

        # calculate ngram predicting loss
        ngram = len(logits_list)
        # print('len of logits_list is ' + str(len(logits_list)))
        expend_targets = targets.new_zeros(ngram, targets.size(0), targets.size(1)).fill_(self.padding_idx)
        for i in range(ngram):
            if i > 0 and self.disable_ngram_loss:
                break
            padding_targets = torch.zeros_like(targets).fill_(self.padding_idx)
            if 'target_idx' in sample:
                expend_targets[i, :, :] = torch.where(sample['target_idx'] >= i, targets, padding_targets)
            else:
                expend_targets[i, :, :] = targets
        targets = expend_targets

        # re-construction loss
        
        rc_loss  = None
        logits = torch.cat(logits_list, dim=0)
        lprobs = F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1, dtype=torch.float32)
        rc_loss = F.nll_loss(
            input=lprobs,
            target=targets.view(-1),
            reduction='sum', ignore_index=self.padding_idx)
        if self.eps > 0.:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_pad_mask = targets.ne(self.padding_idx).view(-1)
            smooth_loss = smooth_loss[non_pad_mask]
            smooth_loss = smooth_loss.sum()
            eps_i = self.eps / lprobs.size(-1)
            # print(smooth_loss.item())
            rc_loss = (1. - self.eps) * rc_loss + eps_i * smooth_loss

        kl_loss, masked_kl_loss = None, None
        if not self.disable_kl_loss and kl is not None:
            kl_loss = kl.clone().sum()
            masked_kl = (kl > self.target_kl)
            masked_kl_loss = torch.mul(kl, masked_kl).sum()
        
        # label classification loss (Empathetic new)    
        class_emotion_loss, class_action_loss = None, None
        emotion_acc, action_acc = None, None
        emotion_acc_count, action_acc_count = None, None
        if self.emotion_labels:
            emotion_lprobs = F.log_softmax(emotion_logits, dim=-1, dtype=torch.float32)
            class_emotion_loss = F.nll_loss(
            input=emotion_lprobs,
            target=sample['emotion'],
            reduction='sum')
            emotion_acc_count = torch.sum(torch.argmax(emotion_lprobs,dim=-1) == sample['emotion'])
        if self.action_labels:
            action_lprobs = F.log_softmax(action_logits, dim=-1, dtype=torch.float32)
            class_action_loss = F.nll_loss(
            input=action_lprobs,
            target=sample['action'],
            reduction='sum')
            action_acc_count = torch.sum(torch.argmax(action_lprobs,dim=-1) == sample['action'])

        # Contrastive loss (#Empathetic new)
        cl_coherence_loss, cl_emotion_loss, cl_action_loss = None ,None ,None
        cl_style_loss = None
        emotion_count, action_count = None, None
        if self.add_cl_loss or self.cl_style:
            if self.cl_coherence:
                cl_coherence_loss = torch.zeros(1).cuda()
            if self.cl_action:
                cl_action_loss = torch.zeros(1).cuda()
                action_count = 0
            if self.cl_emotion:
                cl_emotion_loss = torch.zeros(1).cuda()
                emotion_count = 0
            if self.cl_style:
                cl_style_loss = torch.zeros(1).cuda()
            for i in range(encoder_out['encoder_out'].shape[1]): #暫時用這個替代batch
                if self.cl_action:
                    i_action_list = (sample['action'] ==sample['action'][i]).nonzero().view(-1) #取出同class的indx
                    if len(i_action_list) == 1:
                        pass
                    else:
                        action_count +=1
                        pos_idx = i_action_list[i_action_list != i] # 刪去anchor的idx, i為anchor idx
                        temp_idx = torch.ones(embed_cl_action.shape[0])
                        temp_idx[i_action_list] = 0  # 使得 i_class 在轉bool後皆為False
                        neg_idx= temp_idx.bool()  # 故neg中無pos也無anchor
                        anchor = embed_cl_action[i,:]  # i為anchor idx
                        action_loss = torch.zeros(1).cuda()
                        target = torch.zeros(1).long().cuda()
                        for pos_i in pos_idx:
                            pos_sample = embed_cl_action[pos_i,:].unsqueeze(0)
                            neg_sample = torch.cat((embed_cl_action[neg_idx,:],embed_cl_action_com))
                            samples = torch.cat((pos_sample,neg_sample),dim=0)
                            score = (F.cosine_similarity(anchor,samples)/self.temp_scale_class).unsqueeze(0) # since here anchor and pos_sample are single vector, dim should be 0
                            # print('pos_score is '+ str(pos_score))
                            # print('pos_score type is '+ str(pos_score.type()))
                            # print('pos_score shape is ' + str(pos_score.shape))
                            
                            # neg_score = F.cosine_similarity(anchor,neg_sample)/self.temp_scale
                            # print('neg_score shape is ' + str(neg_score.shape))
                            # print('shape of vector' + str(torch.cat(pos_score,neg_score).shape))
                            action_loss = action_loss +  F.cross_entropy(score,target)
                        cl_action_loss = cl_action_loss + action_loss/len(pos_idx)
                if self.cl_emotion:
                    i_emotion_list = (sample['emotion'] ==sample['emotion'][i]).nonzero().view(-1) #取出同class的indx
                    if len(i_emotion_list) == 1:
                        pass
                    else:
                        emotion_count +=1
                        pos_idx = i_emotion_list[i_emotion_list != i] # 刪去anchor的idx
                        temp_idx = torch.ones(embed_cl_emotion.shape[0])
                        temp_idx[i_emotion_list] = 0  # 使得 i_class 在轉bool後皆為False
                        neg_idx= temp_idx.bool()
                        anchor = embed_cl_emotion[i,:]
                        emotion_loss = torch.zeros(1).cuda()
                        target = torch.zeros(1).long().cuda() # pos sample會在vector的第0個位置，故target設0
                        for pos_i in pos_idx:
                            pos_sample = embed_cl_emotion[pos_i,:].unsqueeze(0)
                            neg_sample = torch.cat((embed_cl_emotion[neg_idx,:],embed_cl_emotion_com))
                            samples = torch.cat((pos_sample,neg_sample),dim=0)
                            score = (F.cosine_similarity(anchor,samples)/self.temp_scale_class).unsqueeze(0) # since here anchor and pos_sample are single vector, dim should be 0
                            # print('pos_score is '+ str(pos_score))
                            # print('pos_score type is '+ str(pos_score.type()))
                            # print('pos_score shape is ' + str(pos_score.shape))
                            
                            # neg_score = F.cosine_similarity(anchor,neg_sample)/self.temp_scale
                            # print('neg_score shape is ' + str(neg_score.shape))
                            # print('shape of vector' + str(torch.cat(pos_score,neg_score).shape))
                            emotion_loss = emotion_loss +  F.cross_entropy(score,target)
                        cl_emotion_loss = cl_emotion_loss + emotion_loss/len(pos_idx)
                if self.cl_coherence:
                    anchor = embed_cl[i,:]
                    temp_idx = torch.arange(embed_cl.shape[0])
                    neg_idx = temp_idx[temp_idx!=i]
                    target = torch.zeros(1).long().cuda() 
                    pos_sample = embed_cl_com[i,:].unsqueeze(0)
                    neg_sample = torch.cat((embed_cl[neg_idx],embed_cl_com[neg_idx]),dim=0) #只要非context,response pair,皆為negative
                    # print('negative samples shape is ' + str(neg_sample.shape))
                    samples = torch.cat((pos_sample,neg_sample),dim=0)
                    # print('samples shape is ' + str(samples.shape))
                    score = (F.cosine_similarity(anchor,samples)/self.temp_scale_instance).unsqueeze(0)
                    # print(' score shape is  ' + str(score.shape))
                    # print(' score is ' + str(score.data))
                    coherence_loss = F.cross_entropy(score,target)
                    cl_coherence_loss = cl_coherence_loss + coherence_loss
                if self.cl_style: 
                    anchor = embed_cl_style_anchor[i,:]
                    temp_idx = torch.arange(embed_cl_style_anchor.shape[0])
                    neg_idx = temp_idx[temp_idx!=i]
                    target = torch.zeros(1).long().cuda() 
                    pos_sample = embed_cl_style_pos[i,:].unsqueeze(0)
                    temp_neg= torch.cat((embed_cl_style_anchor[neg_idx],embed_cl_style_pos[neg_idx]),dim=0) #只要非context,response pair,皆為negative
                    neg_sample= torch.cat((temp_neg,embed_cl_style_com),dim=0)
                    # print('negative samples shape is ' + str(neg_sample.shape))
                    samples_1 = torch.cat((pos_sample,neg_sample),dim=0)
                    samples_2 = torch.cat((anchor.unsqueeze(0),neg_sample),dim=0) #輪流當anchor，故原本的anchor在此做為positive
                    # print('samples shape is ' + str(samples.shape))
                    score_1 = (F.cosine_similarity(anchor,samples_1)/self.temp_scale_instance).unsqueeze(0)
                    score_2 = (F.cosine_similarity(pos_sample,samples_2)/self.temp_scale_instance).unsqueeze(0)
                    # print(' score shape is  ' + str(score.shape))
                    # print(' score is ' + str(score.data))
                    style_loss = F.cross_entropy(score_1,target) + F.cross_entropy(score_2,target)
                    cl_style_loss = cl_style_loss + style_loss
                    
            if cl_coherence_loss is not None:
                cl_coherence_loss = cl_coherence_loss/embed_cl.shape[0]
            if cl_emotion_loss is not None:
                if emotion_count != 0:
                    cl_emotion_loss = cl_emotion_loss / emotion_count
            if cl_action_loss is not None:
                if action_count != 0:
                    cl_action_loss = cl_action_loss / action_count
            if self.cl_style:
                cl_style_loss = cl_style_loss/embed_cl_style_anchor.shape[0]
        
        # total loss
        if self.num_update < 150:
            loss = rc_loss*self.nll_weight
        else:
            loss = rc_loss*self.nll_weight

        if cls_bow_loss is not None:
            loss = loss + self.cls_bow_loss_weight * cls_bow_loss
        if latent_bow_loss is not None:
            loss = loss + self.latent_bow_loss_weight * latent_bow_loss
        if masked_kl_loss is not None:
            loss = loss + self.kl_loss_weight * masked_kl_loss
        if masked_lm_loss is not None:
            loss = loss + self.masked_lm_loss_weight * masked_lm_loss
        
        # # scaling to avoid quickly overfitting on nll loss
        # loss = loss
        
        # Empathetic new
        if class_emotion_loss is not None:  # Empathetic new
            loss = loss + class_emotion_loss * self.class_weight
            # print('use class emotion loss')
        if class_action_loss is not None:  # Empathetic new
            loss = loss + class_action_loss * self.class_weight
            # print('use class action loss')
        if self.num_update > 0:
            # print('self.num_update' + str(self.num_update))
            if cl_action_loss is not None:
                loss = loss + cl_action_loss * self.cl_action_weight
                # print('use cl_action loss')
            if cl_emotion_loss is not None:
                loss = loss + cl_emotion_loss * self.cl_emotion_weight
                # print('use cl_emotion loss')
            if cl_coherence_loss is not None:
                loss = loss + cl_coherence_loss * self.cl_coherence_weight
                # print('use cl coherence loss')
            if cl_style_loss is not None:
                loss = loss + cl_style_loss * self.cl_weight

        sample_size = targets.ne(self.padding_idx).int().sum().item()
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            # 'n_masked_tokens': sample.get('n_masked_tokens', None),
            'rc_loss': utils.item(rc_loss.data) if reduce else rc_loss.data,
        }
        if cls_bow_loss is not None:
            logging_output.update({
                'cls_bow_loss': utils.item(cls_bow_loss.data) if reduce else cls_bow_loss.data})
        if latent_bow_loss is not None:
            logging_output.update({
                'latent_bow_loss': utils.item(latent_bow_loss.data) if reduce else latent_bow_loss.data})
        if masked_kl_loss is not None:
            logging_output.update({
                'masked_kl_loss': utils.item(masked_kl_loss.data) if reduce else masked_kl_loss.data})
        if kl_loss is not None:
            logging_output.update({
                'kl_loss': utils.item(kl_loss.data) if reduce else kl_loss.data})
        if masked_lm_loss is not None:
            logging_output.update({
                'masked_lm_loss': utils.item(masked_lm_loss.data) if reduce else masked_lm_loss.data})
        
        # Empathetic new
        if class_emotion_loss is not None:
            logging_output.update({
                'emotion_loss': utils.item(class_emotion_loss.data) if reduce else class_emotion_loss.data})
        if class_action_loss is not None:
            logging_output.update({
                'action_loss': utils.item(class_action_loss.data) if reduce else class_action_loss.data})
        if emotion_acc_count is not None:
            logging_output.update({'emotion_acc_count': emotion_acc_count.item()})
        if action_acc_count is not None:
            logging_output.update({'action_acc_count': action_acc_count.item()})
        if cl_emotion_loss is not None:
            logging_output.update({
                'cl_emotion_loss': utils.item(cl_emotion_loss.data) if reduce else cl_emotion_loss.data})
            logging_output.update({'emotion_count':emotion_count })
        if cl_action_loss is not None:
            logging_output.update({
                'cl_action_loss': utils.item(cl_action_loss.data) if reduce else cl_action_loss.data})
            logging_output.update({'action_count':action_count })
        if cl_coherence_loss is not None:
            logging_output.update({
                'cl_coherence_loss': utils.item(cl_coherence_loss.data) if reduce else cl_coherence_loss.data})
        if cl_style_loss is not None:
            logging_output.update({
                'cl_style_loss': utils.item(cl_style_loss.data) if reduce else cl_style_loss.data})  
        logging_output.update({'emotion_count': 1})
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):

        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        rc_loss = sum(log.get('rc_loss', 0) for log in logging_outputs)
        cls_bow_loss = sum(log.get('cls_bow_loss', 0) for log in logging_outputs)
        latent_bow_loss = sum(log.get('latent_bow_loss', 0) for log in logging_outputs)
        masked_kl_loss = sum(log.get('masked_kl_loss', 0) for log in logging_outputs)
        kl_loss = sum(log.get('kl_loss', 0) for log in logging_outputs)
        # n_masked_tokens = sum(log.get('n_masked_tokens', 0) for log in logging_outputs)
        # masked_lm_loss = sum(log.get('masked_lm_loss', 0) for log in logging_outputs)
        emotion_loss = sum(log.get('emotion_loss', 0) for log in logging_outputs)
        action_loss = sum(log.get('action_loss', 0) for log in logging_outputs)
        
        emotion_acc_count = sum(log.get('emotion_acc_count', 0) for log in logging_outputs)
        action_acc_count = sum(log.get('action_acc_count', 0) for log in logging_outputs)
        cl_emotion_loss = sum(log.get('cl_emotion_loss', 0) for log in logging_outputs)
        cl_action_loss = sum(log.get('cl_action_loss', 0) for log in logging_outputs)
        cl_coherence_loss = sum(log.get('cl_coherence_loss', 0) for log in logging_outputs)
        cl_style_loss = sum(log.get('cl_style_loss', 0) for log in logging_outputs)
        emotion_count = sum(log.get('emotion_count', 0) for log in logging_outputs)
        action_count = sum(log.get('action_count', 0) for log in logging_outputs)
            
        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'nll_loss': rc_loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            # 'masked_lm_loss': 0 if n_masked_tokens == 0 else masked_lm_loss / n_masked_tokens,
            'rc_loss': rc_loss / sample_size / math.log(2),
            'masked_kl_loss': masked_kl_loss / nsentences,
            'kl_loss': kl_loss / nsentences,
            'latent_bow_loss': latent_bow_loss / sample_size / math.log(2),
            'cls_bow_loss': cls_bow_loss / sample_size / math.log(2),
            'emotion_loss': emotion_loss  / nsentences ,
            'action_loss': action_loss /nsentences,
            'emotion_acc': emotion_acc_count /nsentences,
            'action_acc': action_acc_count /nsentences ,
            'cl_emotion_loss':cl_emotion_loss,
            'cl_action_loss':cl_action_loss,
            'cl_coherence_loss':cl_coherence_loss/nsentences,
            'cl_style_loss':cl_style_loss,
            'emotion_count':emotion_count,
            'action_count':action_count
        }

        return agg_output


# # Test for pretrained model from reddit dataset

# @register_criterion('ngram_language_loss')
# class NgramLmLoss(FairseqCriterion):
#     """
#     Implementation for the loss used in masked language model (MLM) training.
#     """

#     def __init__(self, args, task):
#         super().__init__(args, task)
#         self.eps = args.label_smoothing
#         self.disable_ngram_loss = args.disable_ngram_loss

#     @staticmethod
#     def add_args(parser):
#         """Add criterion-specific arguments to the parser."""
#         # fmt: off
#         parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
#                             help='epsilon for label smoothing, 0 means no label smoothing')
#         parser.add_argument('--disable-ngram-loss', action='store_true',
#                             help='only comput basic stat')
#         # fmt: on

#     def forward(self, model, sample, reduce=True):
#         """Compute the loss for the given sample.
#         Returns a tuple with three elements:
#         1) the loss
#         2) the sample size, which is used as the denominator for the gradient
#         3) logging outputs to display while training
#         """
#         # compute MLM loss
#         logits_list = model(**sample['net_input'], return_all_hiddens=False)[0]
#         targets = model.get_targets(sample, [logits_list[0]])


#         ngram = len(logits_list)
#         # [B, ngram, T]
#         expend_targets = targets.new_zeros(ngram, targets.size(0), targets.size(1)).fill_(self.padding_idx)
#         for i in range(ngram):
#             if i > 0 and self.disable_ngram_loss:
#                 break

#             padding_targets = torch.zeros_like(targets).fill_(self.padding_idx)
#             if 'target_idx' in sample:
#                 expend_targets[i,:,:] = torch.where(sample['target_idx'] >= i, targets, padding_targets)
#             else:
#                 expend_targets[i,:,:] = targets
#         targets = expend_targets

#         logits = torch.cat(logits_list, dim=0) #.view(ngram, *logits_list[0].size())

#         lprobs = F.log_softmax(
#                     logits.view(-1, logits.size(-1)),
#                     dim=-1,
#                     dtype=torch.float32,
#                 )

#         loss = F.nll_loss(
#                lprobs,
#                targets.view(-1),
#                reduction='sum',
#                ignore_index=self.padding_idx,
#                )

#         if self.eps > 0.:
#             smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
#             non_pad_mask = targets.ne(self.padding_idx).view(-1)
#             smooth_loss = smooth_loss[non_pad_mask]
#             smooth_loss = smooth_loss.sum()

#             eps_i = self.eps / lprobs.size(-1)
#             loss = (1. - self.eps) * loss + eps_i * smooth_loss

#         sample_size = targets.ne(self.padding_idx).int().sum().item()

#         logging_output = {
#             'loss': utils.item(loss.data) if reduce else loss.data,
#             'ntokens': sample['ntokens'],
#             'nsentences': sample['nsentences'],
#             'sample_size': sample_size,
#         }
#         return loss, sample_size, logging_output

#     @staticmethod
#     def aggregate_logging_outputs(logging_outputs):
#         """Aggregate logging outputs from data parallel training."""
#         loss = sum(log.get('loss', 0) for log in logging_outputs)
#         ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
#         nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
#         sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

#         agg_output = {
#             'loss': loss / sample_size / math.log(2),
#             'ntokens': ntokens,
#             'nsentences': nsentences,
#             'sample_size': sample_size,
#         }
#         return agg_output