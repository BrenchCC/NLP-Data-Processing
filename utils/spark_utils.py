
import pyspark.sql.functions as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
import datetime
from pyspark import *
import os

# import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix
from collections import defaultdict
import subprocess
from importlib import import_module, reload

#根据最少的类做均衡
def data_balalce(df,f):
    count_info = label_rate(df,f)
    i=0
    for k,v in count_info.items():
        if i==0:
            base_class = k[f]
            base_cnt = v
        elif v< base_cnt:
            base_cnt = v
            base_class = k[f]
        i+=1
    base_df = df.filter(df[f]==base_class)
    for k, v in count_info.items():
        if k[f]!=base_class:
            each_cls = df.filter(df[f]==k[f])
            each_cls_sample = each_cls.sample(fraction=1.0*base_cnt/v,seed = 3)
            base_df = base_df.union(each_cls_sample)
    #
    base_df = base_df.withColumn("rand",F.rand(seed=42))
    base_df = base_df.orderBy(base_df.rand)
    return base_df

# balance_rate: [neg, pos]
# balance_ratio: [neg_ratio, pos_ratio], 如果传入的话, 则不再计算count(), 直接sample
def data_balalce_v2(df, balance_rate=None, label='label', balance_ratio=None):
    if balance_rate is None and balance_ratio is None: return df

    true_sample  = df.filter(df[label] == 1.0)    
    negative_sample  = df.filter(df[label] == 0.0)

    if balance_ratio is None:
        true_sample_n = true_sample.count()
        negative_sample_n = negative_sample.count()
        print('before balance, true sample #: {} and negative sample #:{}'.format(true_sample_n,negative_sample_n))
        negative_rate = balance_rate[0]
        positive_rate = balance_rate[1]
        current_rate = negative_sample_n/true_sample_n
        sample_rate = negative_rate/positive_rate
        if current_rate==sample_rate:
            return df
        elif current_rate>sample_rate:
            howManyTake = int(negative_sample_n*(sample_rate/current_rate))
            negative_sample = negative_sample.sample(fraction=1.0*howManyTake/negative_sample_n,seed = 3)
        else:
            howManyTake = int(true_sample_n*(current_rate/sample_rate))
            true_sample = true_sample.sample(fraction=1.0*howManyTake/true_sample_n,seed = 3)
        print('after balance, true sample #: {} and negative sample #:{}'.format(true_sample.count(),negative_sample.count()))
    else:
        neg_ratio, pos_ratio = balance_ratio
        if neg_ratio != 1: negative_sample = negative_sample.sample(fraction=neg_ratio ,seed = 3)
        if pos_ratio != 1: true_sample = true_sample.sample(fraction=pos_ratio ,seed = 3)

    data_total = true_sample.union(negative_sample)
    # print(data_total.show(5))
    # 拼接后的数据打乱
    data_total = data_total.withColumn("rand",F.rand(seed=42))
    df_rnd = data_total.orderBy(data_total.rand)
    # return data_total
    return df_rnd

def data_sample(df,sample_type,max_n,all_cnt=None):
    data_n = all_cnt if all_cnt is not None else df.count()
    if sample_type == 'sample' and max_n<data_n:
        df = df.sample(fraction=1.0*max_n/data_n,seed = 3)
        return df
    return df
        
def label_rate(df,name):
    out= df.select(name)
    countByValue = out.rdd.countByValue()
    return countByValue
########

    
def merge2list(list_a,list_b):
    # 保持之前的list_a长度不变，并且list_b 中的元素，如果在list_a 中存在，则位置不变。
    # list_a： 旧list；list_b 新list。跟新旧list，并保持长度不变，跟新掉旧list中，没有在新list中出现的元素。
    if list_a[-1] in list_b:
        list_b.remove(list_a[-1])
    last = list_a.pop()
    res = ['']*len(list_a)
    if len(list_a)>=len(list_b):
        for i,x in enumerate(list_a):
            if x in list_b:
                res[i] = x
                list_b.remove(x)
        for i,x in enumerate(res):
            if not x:
                if list_b:
                    res[i] = list_b[0]
                    list_b.pop(0)
                else:
                    res[i] = list_a[i]
    else:
        for i,x in enumerate(list_a):
            if x in list_b:
                res[i] = x
                list_b.remove(x)
        for i,x in enumerate(res):
            if not x:
                res[i] = list_b.pop(0)
    return res+[last]
def update_sparse(df,features,features_info):
    new_features_info = {}
    for f in features:
        countByValue = label_rate(df,f)
        print('-------------------',f)
        pre_list = features_info[f]
        print(len(pre_list))
        print(pre_list)
        sorted_dict = {k: v for k, v in sorted(countByValue.items(), key=lambda item: item[1],reverse = True)}
        sorted_list = [k[f] for k, v in sorted(countByValue.items(), key=lambda item: item[1],reverse = True)]
        print(len(sorted_list))
        print('new_list',sorted_list)
        updated_list = merge2list(pre_list,sorted_list)
        print('after update')
        print(updated_list)
        new_features_info[f] = updated_list
    return new_features_info

##############

def cal_AUC(preds,label):
    # preds: list
    # label: list
    preds = np.array(preds)
    label = np.array(label)
    auc = round(roc_auc_score(label, preds), 4)
    # save log
    txt1 = 'AUC: {}'.format(auc)
    print(txt1)

    th_list = np.linspace(0.0,1.0,21).tolist()
    recall_list = []
    precsion_list = []
    accuracy_list = []
    for th in th_list:
        Y_p = np.where(preds>th, 1,0)
        correct = np.sum((label == Y_p))
        accuracy = round(correct / label.shape[0],4)
        txt2 = 'th: {}, Accuracy:{}'.format(th, accuracy)
        print(txt2)
        tn, fp, fn, tp = confusion_matrix(label, Y_p).ravel()
        precision = tp/float(tp+fp)
        recall = tp/float(tp+fn)

        context1  = 'TP:{}, FP:{}, FN:{}, TN:{}'.format(tp,fp,fn,tn)
        context2 = 'label positive {}, TP: {}, recall:{}'.format((tp+fn), tp, recall)
        context3 = 'label negative {}'.format(fp+tn)
        context4 = 'predict positive {}, TP: {}, precision:{}'.format((tp+fp), tp, precision)
        context5 = 'predict negative {}'.format(fn+tn)
        print(context1)
        print(context2)
        print(context3)
        print(context4)
        print(context5)
        recall_list.append(recall)
        precsion_list.append(precision)
        accuracy_list.append(accuracy)
    # plot
    # plot lines
    plt.figure(figsize=(10,10))
    plt.plot(th_list, recall_list, label = "recall")
    plt.plot(th_list, precsion_list, label = "precision")
    plt.plot(th_list, accuracy_list, label = "accuracy")

    plt.xlabel('threshold')
    # plt.title('model: {}, test data: {}'.format(args.load_from.split('/')[2],DATE[0]))
    # plt.text(0.4, 0.9, 'AUC:{}'.format(auc),size=16)
    plt.xticks(th_list) # add loads of ticks
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) # add loads of ticks
    plt.legend(prop ={'size':16})
    plt.grid()
    plt.show()
    return auc
    # # def isNaN(num):
# #     if float('-inf') < float(num) < float('inf'):
# #         return False 
# #     else:
# #         return True


def dense_analysis(df,f):
    df = df.select(f)
    df_np = df.toPandas().to_numpy()
    # df_np = np.where(np_str=='', '0', np_str).astype('int')
    print(df_np.shape)
    print('min value: {}, max value:{}'.format(np.min(df_np),np.max(df_np)))
    df_np_1 = df_np[df_np>0] # remove zero
    print('median including zero:', np.median(df_np))
    print('median without zero:', np.median(df_np_1))
    print('median: (5, 25, 50, 75, 90, 95,99)')
    print('Quartile including zero:', np.percentile(df_np, (5, 25, 50, 75, 90, 95,99), interpolation='midpoint'))

    if np.any(df_np_1):
        res = np.percentile(df_np_1, (5, 25, 50, 75, 90, 95,99), interpolation='midpoint')
        print('Quartile without zero:',res )
        return res.tolist()
    else:
        print('with zero: ',df_np_1)
        return []


def dense_analysis_gen(df,features):
    print('==============','generate_dense_map','===========================')
    max_n =  10000000
    n = df.count()
    sample = df.sample(fraction=min(1.0,1.0*max_n/n),seed = 3)
    print('raw data: ',n, 'sample data: ', max_n)
    #
    dense_features_info = {}
    dense_features = []
    for f in features:
        print(f,'=============')
        median_value = dense_analysis(sample,f)
        if median_value and median_value[0]< median_value[-1]:
            dense_features_info[f] = {'min':0,'max':median_value[-1]}
            dense_features.append(f)
    return dense_features, dense_features_info


def sparse_analysis_gen(df,features,min_percentage = 0.001):
    print('==============','generate_sparse_map','===========================')
    max_n =  10000000
    n = df.count()
    sample = df.sample(fraction=min(1.0,1.0*max_n/n),seed = 3)
    new_n = sample.count()
    print('raw data: ',n, 'sample data: ', new_n)
    sparse_features_info = {}
    sparse_features = []
    for f in features:
        print('------',f,'-----------')
        curr_list = []
        countByValue = label_rate(sample,f)
        sorted_dict = {k: v for k, v in sorted(countByValue.items(), key=lambda item: item[1],reverse = True)}
        sorted_list = [[k[f],v] for k, v in sorted(countByValue.items(), key=lambda item: item[1],reverse = True)]
        #
        for i, x in enumerate(sorted_list):
            if float(x[1]/new_n)>=min_percentage:
                print('top: {}'.format(i),x[0],x[1],float(x[1]/new_n))
                curr_list.append(x[0])
        if 2<=len(curr_list)<=100:
            # 选取有效的特征
            sparse_features_info[f] = curr_list
            sparse_features.append(f)
    #
    print(sparse_features_info)
    return sparse_features, sparse_features_info

def generate_dict(df, features, feature_type = 'dense', gen_map = True, src_map = None):
    #feature_type : dense. sparse
    res = {}
    if feature_type =='dense':
        if gen_map:
            features_new, res = dense_analysis_gen(df,features)
        else:
            print('copy dict from {}'.format(src_map))
            features_new, res = features, src_map
        return features_new,res
    elif feature_type =='sparse':
        if gen_map:
            features_new, res = sparse_analysis_gen(df,features)
        else:
            print('copy dict from {}'.format(src_map))
            features_new, res = features, src_map
        return features_new, res


def balance_and_sample(df_s,balance='unbalance',rate =[5,1],sample_type = 'sample',max_n=2000000, label='label', print_cnt=False, balance_ratio=None, all_cnt=None):
    # print(df_s.count(),label_rate(df_s,label))
    # step1: balance data
    print('[1] balance data...')
    if balance == 'unbalance': rate = None
    df_s = data_balalce_v2(df_s, balance_rate=rate, label=label, balance_ratio=balance_ratio)
    if print_cnt:
        print(balance,df_s.count())
        print(balance,label_rate(df_s,label))
    # step2: sample data
    print('[2] sample data...')
    df_sample = data_sample(df_s, sample_type, max_n, all_cnt=all_cnt)
    if print_cnt:
        print(sample_type,df_sample.count())
        print(sample_type,label_rate(df_sample,label))
    return df_sample

def generate_train_test(df_s,balance='unbalance',rate =[5,1],sample_type = 'sample',max_n=2000000, label='label', split_rate=[0.8,0.2], print_cnt=False, balance_ratio=None, all_cnt=None):
    df_sample = balance_and_sample(df_s,
                                    balance=balance,
                                    rate=rate,
                                    sample_type=sample_type,
                                    max_n=max_n,
                                    label=label,
                                    print_cnt=print_cnt,
                                    balance_ratio=balance_ratio,
                                    all_cnt=all_cnt)
    # step3: split data for traing, evluation and testing
    print('[3] split data...')
    train_data, test_data = df_sample.randomSplit(split_rate, seed=3)
    # print('train data: {}, test data:{}'.format(train_data.count(),test_data.count()))
    ##################################################
    return train_data, test_data


def save(df_s,hdfs_path,save_path,balance='unbalance',rate=[1,1], sample_type='sample', max_n=2000000, label='label', split_rate=[0.8,0.2], print_cnt=False, balance_ratio=None, all_cnt=None, partition=-1):
    print(f'Columns: {df_s.columns}')
    train_data, test_data = generate_train_test(df_s,
                                                balance=balance,
                                                rate=rate,
                                                sample_type=sample_type,
                                                max_n=max_n,
                                                label=label,
                                                split_rate=split_rate,
                                                print_cnt=print_cnt,
                                                balance_ratio=balance_ratio,
                                                all_cnt=all_cnt
                                                )
    ##################################################
    train_parquet_hdfs=hdfs_path+save_path+'_train.parquet'
    print(f'Writing data to {train_parquet_hdfs}')
    if partition > 0: train_data = train_data.coalesce(partition)
    train_data.write.mode('overwrite').option("header", "true").format('parquet').save(train_parquet_hdfs)

    test_parquet_hdfs=hdfs_path+save_path+'_test.parquet'
    print(f'Writing data to {test_parquet_hdfs}')
    if partition > 0: test_data = test_data.coalesce(partition)
    test_data.write.mode('overwrite').option("header", "true").format('parquet').save(test_parquet_hdfs)
    
    if print_cnt:
        print('train date:',(train_data.count(), len(train_data.columns)))
        print('label rate:',label_rate(train_data,label))
        print('test date:',(test_data.count(), len(test_data.columns)))
        print('label rate:',label_rate(test_data,label))
    tmp = train_data.take(3)[0]
    print(tmp)
    print('model input length: ',len(tmp['input_data']))
    print('train_path:',train_parquet_hdfs)
    print('test_path:',test_parquet_hdfs)


def save_all(df_s,hdfs_path,save_path,balance='unbalance',rate=[1,1], sample_type='sample', max_n=2000000, label='label', print_cnt=False, balance_ratio=None, all_cnt=None):
    print(f'Columns: {df_s.columns}')
    df_sample = balance_and_sample(df_s,
                                    balance=balance,
                                    rate=rate,
                                    sample_type=sample_type,
                                    max_n=max_n,
                                    label=label,
                                    print_cnt=print_cnt,
                                    balance_ratio=balance_ratio,
                                    all_cnt=all_cnt)
    ##################################################
    all_parquet_hdfs=hdfs_path+save_path+'_all.parquet'
    print(f'Writing data to {all_parquet_hdfs}')
    df_sample.write.mode('overwrite').option("header", "true").format('parquet').save(all_parquet_hdfs)

    print('all_parquet_hdfs:',all_parquet_hdfs)
    return all_parquet_hdfs


#.  analysis toutiao_exit_predict###############################################################
def penetration_analysis_rdd(df_rdd,pet_list = ['did','session_id'],feature='name', target_value='huoshan_video_show'):
    # 分析某个特张值的滲透率,did,session_id
    raw_n = df_rdd.count()
    df_rdd_s = df_rdd.flatMap(lambda x: [x] if x[feature]==target_value else [])
    select_n = df_rdd_s.count()
    print('raw_n: {}, select_n: {}, penetration:{}'.format(raw_n,select_n,select_n/raw_n))
    for pet in pet_list:
        print('penetration name: {}'.format(pet))
        pet_n = len(df_rdd.map(lambda x: x[pet]).countByValue())
        select_pet_n = len(df_rdd_s.map(lambda x: x[pet]).countByValue())
        print('pet_n: {}, select_pet_n: {}, penetration:{}'.format(pet_n,select_pet_n,select_pet_n/pet_n))


def session_id_count(df,id_list = False):
    session_id_info = label_rate(df,'session_id')
    if not id_list:
        return len(session_id_info), []
    else:
        session_id = []
        for k,v in session_id_info.items():
            session_id.append(k['session_id'])
        return len(session_id_info),session_id

def count_by_name(df=None,name=None,suffix=None):
    unique_uids = [row[name] for row in df.select(name).distinct().collect()]
    n = len(unique_uids)
    if suffix: print('name: {}, {}: '.format(name,suffix),n)
    return unique_uids

    
def did_count(df,id_list = False):
    did_info = label_rate(df,'did')
    if id_list:
        did = []
        for k,v in did_info.items():
            did.append(k['did'])
        return len(did_info),did
    else:
        return len(did_info),[]
def unique_did(error_did,right_did):
    return len(list(set(error_did) - set(right_did)))

def analysisByError_code(df,error_code,total_pred_cnt,total_did_cnt,right_did):
    # total
    df_e = df.filter((df['error_code']==error_code))
    pred_cnt = df_e.count()
    did_cnt, error_did = did_count(df_e, True)
    print('预测次数,  总的预测次数:{},该error的预测次数,{},占比:{}'.format(total_pred_cnt,pred_cnt,float(pred_cnt)/float(total_pred_cnt)))
    print('did个数,  总的did个数:{},该error的did个数,{},占比:{}'.format(total_did_cnt,did_cnt,float(did_cnt)/float(total_did_cnt)))
    cnt = unique_did(error_did,right_did)
    print('该error，导致多少did没有被推理:{},占比:{}'.format(cnt,float(cnt)/total_did_cnt))

def pred_and_did(df):
    pred_cnt = df.count()
    did_cnt, did_list = did_count(df, True)
    return pred_cnt,did_cnt,did_list

def analysis_toutiao_exit_predict(df,total_data_did):
    total_pred_cnt, total_did_cnt, _ = pred_and_did(df)
    print('total predict:{}, total predict did:{}'.format(total_pred_cnt,total_did_cnt))
    print('所有数据的did: {}'.format(total_data_did))
    # 有效预测
    print('-----------有效预测----------------')
    right = df.filter(df['error_code']==0)
    right_pred_cnt, right_did_cnt, right_did = pred_and_did(right)
    print('有效预测个数:{}, 占比:{}'.format(right_pred_cnt,right_pred_cnt/total_pred_cnt))
    print('有效预测did个数: {}, 占比:{}'.format(right_did_cnt,right_did_cnt/total_did_cnt))
    print('有效预测did个数: {}, 占所有数据的比:{}'.format(right_did_cnt,right_did_cnt/total_data_did))
    # 无效预测
    error = df.filter(df['error_code']>0)
    error_pred_cnt, error_did_cnt, error_did = pred_and_did(error)
    print('-----------无效预测----------------')
    print('无效预测个数:{}, 占比:{}'.format(error_pred_cnt,error_pred_cnt/total_pred_cnt))
    print('出现过无效预测的did个数: {}, 占比:{}'.format(error_did_cnt,error_did_cnt/total_did_cnt))
    unique_did_cnt = unique_did(error_did,right_did)
    print('error 导致没有推理的did个数:{},占比:{}'.format(unique_did_cnt,unique_did_cnt/total_did_cnt))

    error_info = label_rate(error,'error_code')
    for k,v in error_info.items():
        print('------error_code:{}-----'.format(k))
        print('error_code: {}, count:{}, rate:{}'.format(k,v,v/total_pred_cnt))
        analysisByError_code(df,k['error_code'],total_pred_cnt,total_did_cnt, right_did)
    
def plot_penetration(df,label_name,total_data_did):
    total_pred_cnt, total_did_cnt, _ = pred_and_did(df) 
    print('total predict cnt:{}, total did:{}'.format(total_pred_cnt,total_did_cnt))
    print('所有数据的did: {}'.format(total_data_did))
    th_list = np.linspace(0.0,1.0,21).tolist()
    pred_penetration = []
    did_penetration = []
    did_penetration_a = []
    for th in th_list:
        th = round(th,4)
        df_th = df.filter(df['predict']>=th)
        th_pred_cnt, th_did_cnt, _ = pred_and_did(df_th)
        pred_p = round(th_pred_cnt/total_pred_cnt,5)
        did_p = round(th_did_cnt/total_did_cnt,5)
        did_p_a = round(th_did_cnt/total_data_did,5)
        pred_penetration.append(pred_p)
        did_penetration.append(did_p)
        did_penetration_a.append(did_p_a)
        print('th:{},pred_cnt:{},percentage:{},did_cnt:{},did_percentage:{}'.format(th,th_pred_cnt,pred_p,th_did_cnt,did_p))
        print('th:{},pred_cnt:{},percentage:{},did_cnt:{},did_percentage_a:{}'.format(th,th_pred_cnt,pred_p,th_did_cnt,did_p_a))
    # plot lines
    plt.figure(figsize=(10,10))
    plt.plot(th_list, pred_penetration, label = "pred_penetration")
    plt.plot(th_list, did_penetration, label = "did_penetration")
    plt.plot(th_list, did_penetration_a, label = "did_penetration_all_user")

    plt.xlabel(label_name)
    plt.legend(prop ={'size':16})
    plt.grid()
    plt.show()

# '''
# #.  filter related function #########################################
# '''
def filter_balance_by_event_name(df,inference_event,balance,rate,label):
    #  first
    print(inference_event[0],'----')
    df_selected = df.filter(df['event_name']==inference_event[0])
    df_filter = data_balalce_v2(df_selected, rate=rate,label = label)
    # 
    for f in inference_event[1:]:
        print(f,'-----')
        df_selected = df.filter(df['event_name']==f)
        if len(label_rate(df_selected,label))>1:
            tmp = data_balalce_v2(df_selected,rate=rate,label=label)
            df_filter = df_filter.union(tmp)
        else:
            df_selected = df_selected.withColumn("rand",F.rand(seed=42))
            df_filter = df_filter.union(df_selected)
    return df_filter

def filter_balance_by_total(df,inference_event,balance,rate,label):
    #  first
    print(inference_event[0],'----')
    df_filter = df.filter(df['event_name']==inference_event[0])
    # 
    for f in inference_event[1:]:
        print(f,'-----')
        df_selected = df.filter(df['event_name']==f)
        df_filter = df_filter.union(df_selected)
    df_filter = data_balalce_v2(df_filter, rate=rate,label = label)
    #
    return df_filter

def filter_by_min(df,feature,min_value):
    df_tmp = df.filter(df[feature]<min_value)
    count_info = label_rate(df_tmp,'session_id')
    print('error session:',len(count_info))
    sess_s = []
    for k,v in count_info.items():
        sess_s.append(k['session_id'])
    df_s = df.filter(~df.session_id.isin(sess_s))
    return df_s
def filter_by_max(df,feature,max_value):
    df_tmp = df.filter(df[feature]>max_value)
    count_info = label_rate(df_tmp,'session_id')
    print('error session:',len(count_info))
    sess_s = []
    for k,v in count_info.items():
        sess_s.append(k['session_id'])
    df_s = df.filter(~df.session_id.isin(sess_s))
    return df_s

def filter_by_name(df,feature,target_value):
    # # 选取具有某个特征的session 数据
    df_tmp = df.filter(df[feature]==target_value)
    count_info = label_rate(df_tmp,'session_id')
    print('availbel session:',len(count_info))
    sess_s = []
    for k,v in count_info.items():
        sess_s.append(k['session_id'])
    df_s = df.filter(df.session_id.isin(sess_s))
    return df_s
# '''
# #.  根据不同的 event_name 分析， label rate#########################################
# '''

def analysis_labels(df, label_list):
    count_info = label_rate(df,'event_name')
    event_name_list = []
    for k,v in count_info.items():
        event_name_list.append(k)
    for f in label_list:
        print(label_rate(df,f))
    for f in event_name_list:
        print('------',f['event_name'],count_info[f])
        df_selected  = df.filter((df['event_name']== f['event_name']))
        for l in label_list:
            print(df_selected.count(),label_rate(df_selected,l))

# '''
# #.  根据是否有enter_app， 选数据集#########################################
# '''

#DATA_TYPE = 'mix'     # with_enter_app, without_enter_app, mix
def select_enter_app(df,data_type):
    if data_type =='mix':
        df_filter = df
        print('df_mix:', df_filter.count())
    elif data_type =='with_enter_app':
        df_filter = df.filter(df['enter_app_info']==True)
        print('df_with_enter_app:', df_filter.count())
    elif data_type =='without_enter_app':
        df_filter = df.filter(df['enter_app_info']==False)
        print('df_without_enter_app:', df_filter.count())
    else:
        print('DATA_TYPE Error')
        df_filter = df
        print('df_mix:', df_filter.count())
    return df_filter

# '''
# #.  根据是是新用户， 选数据集#########################################
# '''

# '1441316838565839423','1441154329194859489',
#                         '1441400840757033267'
def select_is_newuser(df,is_newuser):
    count_info = label_rate(df,'is_newuser')
    print(count_info)

    hash_is_newuser =['1441316838565839423','1441154329194859489','1441400840757033267']
    is_newuser_list = []
    for k,v in count_info.items():
        is_newuser_list.append(k)
    for k in is_newuser_list:
        if k not in is_newuser_list:
            print('hash_is_newuser has changed')
            return None
    print('hash_is_newuser is right')
    if is_newuser==0:
        df_filter = df.filter(df['is_newuser']=='1441316838565839423')
        print('is_newuser: {}, old user, {}'.format(is_newuser,df_filter.count()))
    elif is_newuser==1:
        df_filter = df.filter(df['is_newuser']=='1441154329194859489')
        print('is_newuser: {}, new user, {}'.format(is_newuser,df_filter.count()))
    elif is_newuser==2:
        df_filter = df.filter(df['is_newuser']=='1441400840757033267')
        print('is_newuser: {}, value is None, {}'.format(is_newuser,df_filter.count()))
    else:
        df_filter = df
        print('is_newuser: {}, all user, {}'.format(is_newuser,df_filter.count()))
    return df_filter

# '''
# #.  根据决策树分析特征的重要性#########################################
# '''
def DT(df_pd,input_features,label='label'):
    X = df_pd[input_features]
    Y = df_pd[label]
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
    # clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train,y_train)
    #
    importances = sorted([*zip(input_features,clf.feature_importances_)],key = lambda s: s[1],reverse = True)
    y_test_hat = clf.predict(x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_hat).ravel()
    precision = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    return clf,importances,tn, fp, fn, tp,precision,recall

def importance_analysis(df_str2float,balance,rate,sample_type,max_n,input_features,label='label'):
    print('before balance:',label_rate(df_str2float,label))
    df_balance = data_balalce_v2(df_str2float,rate=rate,label=label)
    print('after balance:',label_rate(df_balance,label))
    df_sample = data_sample(df_balance,sample_type,max_n)
    print('after sample:',label_rate(df_sample,label))
    print(sample_type,df_sample.count())
    print(df_sample.take(1)[0])
    df_pd = df_sample.toPandas() 
    #
    clf,importances,tn, fp, fn, tp,precision,recall = DT(df_pd,input_features,label)
    print('precision: {},recall:{}'.format(round(precision,4),round(recall,4)))
    print('importances:')
    for x in importances:
        print(x)

# ----------------新的产生MAP的函数----------------#
# 遍历df的所有row, 对features里的每个值(df的column), 计算其对应的value_cnt
def count_by_values(df, features):
    def seqOp(accu, row):
        for key in features:
            val = row[key]
            accu[key][val] += 1
        return accu

    def combOp(accu1, accu2):
        for key in features:
            key_map1 = accu1[key]
            key_map2 = accu2[key]
            for k in key_map2:
                key_map1[k] += key_map2[k]
        return accu1

    init_val = {key: defaultdict(int) for key in features}
    return df.aggregate(init_val, seqOp, combOp)

def genSparseFeatures(df, features, count_group_size=1000000,min_percentage = 0.001,max_len =100):
    print('==============','generate_sparse_map','===========================')
    print(f'min_percentage: {min_percentage}, max_len: {max_len}')
    max_n=10000000
    n = df.count()

    fraction = 1.0 * max_n / n
    if fraction < 1:
        sample = df.sample(fraction=min(1.0,1.0*max_n/n),seed = 3)
        new_n = sample.count()
    else:
        sample = df
        new_n = n
    print(f'raw data: {n}, sample data: {new_n}')

    print(f'Calculating frequency, columns: {len(features)}')

    # 判断哪些特征不在数据集中:
    df_columns = list(df.columns)
    print('Nonexistent features: ',[x for x in features if x not in df_columns])
    features = [x for x in features if x in df_columns]
    
    # 防止oom, 每50个feature运行一次
    idx = 0
    all_map = {}
    while idx < len(features):
        sub_map = count_by_values(sample.rdd, features[idx:idx+count_group_size])
        # for k in sub_map: print(k, len(sub_map[k]))
        all_map.update(sub_map)
        idx += count_group_size

    new_features = []
    new_features_info = {}
    for f in features:
        print('-------features-------:{}'.format(f))

        sorted_list = [[k,v] for k, v in sorted(all_map[f].items(), key=lambda item: item[1],reverse = True)]
        curr_list =[]
        for i, x in enumerate(sorted_list):
            if float(x[1]/new_n)>=min_percentage:
                print('top: {},{},{},{}'.format(i, x[0],x[1],float(x[1]/new_n)))
                curr_list.append(x[0])
        if 2<=len(curr_list)<=max_len:
            # 选取有效的特征
            new_features_info[f] = curr_list
            new_features.append(f)
    return new_features, new_features_info

def genDenseFeatures(df, features, count_group_size=1000000):
    print('==============','generate_dense_map','===========================')
    max_n =  10000000
    n = df.count()
    sample = df.sample(fraction=min(1.0,1.0*max_n/n),seed = 3)
    # print('raw data: ',n, 'sample data: ', sample.count())
    # 判断哪些特征不在数据集中:
    df_columns = list(df.columns)
    print('Nonexistent features: ',[x for x in features if x not in df_columns])
    features = [x for x in features if x in df_columns]
    # 防止oom, 每50个feature运行一次
    idx = 0
    median_value_all = []
    while idx < len(features):
        median_value_list = sample.approxQuantile(features[idx:idx+count_group_size], [0.1,0.95], 0.01) #95分为数，更合理一点
        median_value_all.extend(median_value_list)
        idx += count_group_size
    # median_value_all = sample.approxQuantile(features,[0.1,0.99],0.01)
    #
    dense_features_info = {}
    dense_features = []
    for i, f in enumerate(features):
        median_value = median_value_all[i]
        print('{},{}'.format(f,median_value))
        if median_value and median_value[0]< median_value[-1]:
            dense_features_info[f] = {'min':0,'max':median_value[-1]}
            dense_features.append(f)
    return dense_features, dense_features_info
    #---------------------------------------------------------#

def print_features(dense_f,dense_f_info,sparse_f,sparse_f_info):
    d_n = 0
    if dense_f:
        print('----------dense-------------------')
        for f in dense_f:
            print('{}: {}'.format(f,[dense_f_info[f]['min'],dense_f_info[f]['max']]))
        d_n = len(dense_f)
    #
    s_n = 0
    if sparse_f:
        print('----------sparse-------------------')
        for f in sparse_f:
            s_n+= len(sparse_f_info[f])
            print('{}: len:{}'.format(f,len(sparse_f_info[f])))
    total_n = d_n + s_n
    print('total_n: {}, dense_n: {}, sparse_n: {}'.format(total_n,d_n,s_n))

def getdate(current_day='20230425', N = 0):
    '''
    N为负数时，前abs(N)天
    N为正数时，后N天
    DATE_d: '20230425'
    '''
    if N>0:
        DATE_d = (datetime.datetime.strptime(current_day, '%Y%m%d') + datetime.timedelta(days = N)).strftime("%Y%m%d")
    else:
        DATE_d = (datetime.datetime.strptime(current_day, '%Y%m%d') - datetime.timedelta(days = abs(N))).strftime("%Y%m%d")
    return DATE_d
############## HDFS 路径相关 ########################
def date_duration2date_list(s_date='20230428',e_date='20230428'):
    '''
    s_date: 开始日期
    e_date: 结束日期
    res: 两个日期之间的所有日期
    '''
    start_date = datetime.datetime.strptime(s_date, '%Y%m%d')
    end_date = datetime.datetime.strptime(e_date, '%Y%m%d')
    if start_date>end_date:
        return []
    res = []
    while start_date <= end_date:
        # print(start_date.strftime("%Y%m%d"))
        res.append(start_date.strftime("%Y%m%d"))
        start_date += datetime.timedelta(days = 1)
    return res

def generate_path_list(s_date='20230428',e_date='20230428',comment_path = '', special_path = ''):
    date_list = date_duration2date_list(s_date=s_date,e_date=e_date)
    res = []
    if not date_list: return res
    for x in date_list:
        res.append(os.path.join(comment_path,x,special_path))
    return res, date_list
# hdfs 文件处理相关
def get_map_from_hdfs(hdfs_path=None):
    # 复制到本地临时文件
    os.system('hdfs dfs -get {} {}'.format(hdfs_path,'temp_data_features.py'))
    # 导入临时文件
    import temp_data_features as idf
    # 解析导入文件
    my_names = [name for name in dir(idf) if not name.startswith('__')]
    res = {}
    # 获取每个变量的值
    print('[form .py to dict]: ')
    for name in my_names:
        value = getattr(idf, name)
        res[name]=value
    # 返回data_features 为字典格式
    print('current_data_features kyes: ',res.keys())
    for k,v in res.items():
        if '_len' in k:
            print('{}:{}'.format(k,v))
    return res

def uploadHDFS(res_path, dst_path):
    os.system('hdfs dfs -put {} {}'.format(res_path,dst_path))
def downloadHDFS(src_path=None, dst_path=None):
    os.system('hdfs dfs -get {} {}'.format(src_path,dst_path))
    return dst_path
def deleteHDFS(dst_path):
    os.system('hdfs dfs -rm -r {}'.format(dst_path))

def get_new_map_from_hdfs(hdfs_map_path=None, local_map_file=None):
    # local_map_file， 必须是 module_name.py 的格式
    if '.py' not in local_map_file:
        print('local_map_file: ',local_map_file)
        raise RuntimeError("local_map_file Error")
    try:
        # 尝试删除文件
        os.remove(local_map_file)
        print(f"local 文件 {local_map_file} 已删除")
    except OSError as e:
        # 如果删除失败，抛出异常
        print(f"删除文件时发生错误: {e}")
    print('download from hdfs map: ',hdfs_map_path)
    local_map = downloadHDFS(src_path=hdfs_map_path, dst_path=local_map_file)
    print('local map path: ', local_map)
    # 判断MAP是否存在
    new_map = {}
    if os.path.exists(local_map):
        module_name = local_map[:-3]
        print('module_name: ',module_name)
        # import _current_data_features as idf
        idf = import_module(module_name) # 第一次导入模块
        reload(idf)
        print('from _data_features.py to dict')
        new_map = module2dict(idf)
    return new_map

def get_namelist_from_HDFS(path):
    try:
        cmd = 'hdfs dfs -ls '+path
        nameList = []
        files = subprocess.check_output(cmd, shell=True).decode("utf-8")
        for x in files.strip().split('\n'):
            for name in x.split():
                if 'hdfs:' in name:
                    nameList.append(name)
        return nameList
    except:
        return []
def check_hdfs_path_is_or_not_in_hdfs(father_path=None, file_path=None):
    # 核实hdfs文件是否在父目录下面
    path_list = get_namelist_from_HDFS(father_path)
    if file_path in path_list:
        return True
    else:
        return False

# hdfs 文件处理相关 
def get_saved_path(hdfs_path, suffix='', name = ''):
    res = hdfs_path+suffix
    print("{:<35}".format('[{}]:'.format(name)),suffix)
    return res
def check_duration_hdfs_path(s_date='20230428',e_date='20230428',comment_path = '', special_path = ''):
    date_path_list, date_list = generate_path_list(s_date=s_date,e_date=e_date,comment_path = comment_path, special_path = special_path)
    all_name_list = get_namelist_from_HDFS(comment_path)
    all_name_list = [os.path.join(x,special_path) for x in all_name_list]
    n_move = len('hdfs://haruna/home/byte_client_infra_pitaya/quit_predict/')
    print('[check data]: start-{}, end-{}, {}'.format(s_date,e_date,special_path))
    is_ready = True
    for path in date_path_list:
        if not path in all_name_list:
            is_ready = False
            print("{:<10}".format('No exist:'),path[n_move:])
        # else:
        #     print("{:<10}".format('Exist:'),path[n_move:])
    print('[check data result]: {}'.format(is_ready))
    return is_ready
# ###############################################
def count_info_analysis(count_info,f):
    print(count_info)
    cnt = 0
    for k,v in count_info.items():
        cnt+=v
    for k,v in count_info.items():
        if k[f]=='0':
            print('模型正常推理:        {}, rate: {}'.format(v,round(v/cnt,5)))
        if k[f]=='401':
            print('长度不满足要求:      {}, rate: {}'.format(v,round(v/cnt,5)))
        if k[f]=='402':
            print('直播数据，没参与重排: {}, rate: {}'.format(v,round(v/cnt,5)))
        if k[f]=='403':
            print('广告数据，没参与重排: {}, rate: {}'.format(v,round(v/cnt,5)))
        if k[f]=='1000':
            print('退出session的个数: {}, rate: {}'.format(v,round(v/cnt,5)))
def count_info_top(count_info,f,top_n):
    cnt = 0
    for k,v in count_info.items():
        cnt+=v
    sorted_list = [[k[f],v] for k, v in sorted(count_info.items(), key=lambda item: item[1],reverse = True)]
    for k,v in sorted_list[:top_n]:
        print('{}, {}, rate: {}'.format(k,v,round(v/cnt,5)))
def print_cloud_data(df,sparse_name = None,dense_name = None):
    # check 云端数据
    # sparse_name= ['g_source','group_source_c','recent_act_staytimes','list_entrance', 'new_source',
    #                 'category_name_native', 'article_type_native','enter_from_native']
    # dense_name = ['publish_time2current','impression_count','digg_count_c','_DISLIKE','video_finish_diff_l3']
    if sparse_name:
        for f in sparse_name:
            print('----sparse: {} -------'.format(f))
            count_info_top(label_rate(df,f),f,top_n=10)
    if dense_name:
        for f in dense_name:
            print('----dense: {} -------'.format(f))
            print(dense_analysis(df,f))
# #################################################################################
def module2dict(m):
    # type: <class 'module'>
    my_names = [name for name in dir(m) if not name.startswith('__')]
    res = {}
    # 获取每个变量的值
    print('[form .py to dict]: ')
    for name in my_names:
        value = getattr(m, name)
        res[name]=value
    return res
def check_local_data_features():
    # 判断MAP是否存在
    old_map = {}
    if os.path.exists('_data_features.py'):
        import _data_features as idf
        print('from _data_features.py to dict')
        old_map = module2dict(idf)
        print(old_map.keys())
    if old_map: 
        is_exist = True
    else:
        is_exist = False
    return old_map, is_exist

def check_key_is_or_not_in_dict(new_map=None, ckeck_list=None):
    # 核实 ckeck_list 中的元素是否为 new_map 中的key
    print('new_map kyes: ',new_map.keys())
    for f in ckeck_list:
        if f in new_map.keys():
            print(f,new_map[f])
        else:
            print('{} not in map'.format(f))

def save_new_map2hdfs(new_map_file,map_dict,date_set_path):
    # 对应于Window_N天的训练集数据的新MAP
    with open(new_map_file, 'w') as f:
        for key, value in map_dict.items():
            f.write('{} = {}\n'.format(key, value))
        print('Write new map Done: {}'.format(new_map_file))
    # 如果文件存在，会写失败
    uploadHDFS(res_path=new_map_file, dst_path=date_set_path)
    print('new_map_path: ', date_set_path+new_map_file)
#####################################
