import numpy as np


class RejectAcceptSampling:
    """
    拒绝-接受采样工具类，支持一维和二维条件分布的采样，用于从目标条件分布 p(条件变量|给定变量) 中获取样本。
    核心思想是通过提议分布 q 并结合缩放常数 M，以一定概率接受或拒绝采样得到的样本，从而逼近目标分布。
    """
    def __init__(self, dimensionality=1):
        """
        初始化拒绝-接受采样器
        :param dimensionality: 采样的维度，目前支持 1（一维）、2（二维），用于区分不同维度的采样逻辑，默认 1 维
        """
        self.dimensionality = dimensionality

    def rejection_sampling_1D(scene, NDE_intersection, 
                              dim=[0, 1, 2, 3, 4, 5, 6, 7], 
                              max_trials=1000, random_state=None, type='sample'):
        """
        一维条件分布的拒绝-接受采样：从 p(a|s) = p(a,s)/p(s) 中采样，即已知 s 时，采满足条件的 a
        """
        # 初始化随机数生成，若有指定随机种子则固定，保证可复现性
        np.random.seed(random_state)  
        accept=0
        for trial_idx in range(max_trials):
            # 步骤1：从提议分布 q 中采样获取候选的 a
            while True:
                accelerate = NDE_intersection['proposal_accelerate'].sample()
                if 0 <= accelerate <= 1:
                    log_proposal_accelerate = NDE_intersection['proposal_accelerate'].logpdf(accelerate)
                    break
            scene_behavior = np.concatenate([scene[0], accelerate])
            # 步骤2：计算联合分布和边缘分布的对数概率密度
            log_p_as_val = NDE_intersection['scene_behavior'].logpdf(scene_behavior.reshape(1,-1))  
            log_p_s_val  = NDE_intersection['scene_behavior'].logpdf_marginal(scene, dim)  
            
            # 步骤3：计算条件分布的对数概率密度
            # 条件概率公式：p(a,b|s) = p(a,b,s)/p(s)，取对数后为 log(p(a,b|s)) = log(p(a,b,s)) - log(p(s))
            log_p_cond_val = log_p_as_val - log_p_s_val  
            
            # 步骤4：计算接受概率的对数形式
            # 接受概率 alpha = p(a,b|s) / (M*q(a)*q(b))，取对数后 log(alpha) = log(p(a,b|s)) - (log(M) + log(q(a)) + log(q(b)))
            log_alpha = log_p_cond_val - (np.log(NDE_intersection['proposal_M']) + log_proposal_accelerate )  
            
            # 步骤5：判断是否接受该样本，增加对 log_p_cond_val 过大情况的处理
            # if log_p_cond_val > 10 or np.log(np.random.rand()) < log_alpha:  
            if np.log(np.random.rand()) < log_alpha:  
                if type=='sample':
                    return scene_behavior, log_p_cond_val, trial_idx + 1  
                else:
                    accept+=1
                
        if type=='sample':
            print('无法采样')            
            pass
        else:
            return accept/max_trials

    def rejection_sampling_2D(scene, NDE_intersection, 
                              dim=[0, 1, 2, 3, 4, 5, 6, 7], 
                              max_trials=1000, random_state=None, type='sample'):
        """
        二维条件分布的拒绝-接受采样：从 p(a,b|s) = p(a,b,s)/p(s) 中采样，即已知 scene时，采满足条件的 a、b

        """
        # 初始化随机数生成，若有指定随机种子则固定，保证可复现性
        np.random.seed(random_state)  
        accept=0
        for trial_idx in range(max_trials):
            # 步骤1：从提议分布中采样二维behavior，需满足 0<=behavior<=1 的约束
            while True:
                accelerate = NDE_intersection['proposal_accelerate'].sample()
                if 0 <= accelerate <= 1:
                    log_proposal_accelerate = NDE_intersection['proposal_accelerate'].logpdf(accelerate)
                    break
            while True:
                steering_angle = NDE_intersection['proposal_steering_angle'].sample()
                if 0 <= steering_angle <= 1:
                    log_proposal_steering_angle = NDE_intersection['proposal_steering_angle'].logpdf(steering_angle)
                    break
            scene_behavior = np.concatenate([scene[0], accelerate, steering_angle])
            
            # 步骤2：计算联合分布和边缘分布的对数概率密度
            log_p_as_val = NDE_intersection['scene_behavior'].logpdf(scene_behavior.reshape(1,-1))  
            log_p_s_val = NDE_intersection['scene_behavior'].logpdf_marginal(scene, dim)  
            
            # 步骤3：计算条件分布的对数概率密度
            # 条件概率公式：p(a,b|s) = p(a,b,s)/p(s)，取对数后为 log(p(a,b|s)) = log(p(a,b,s)) - log(p(s))
            log_p_cond_val = log_p_as_val - log_p_s_val  
            
            # 步骤4：计算接受概率的对数形式
            # 接受概率 alpha = p(a,b|s) / (M*q(a)*q(b))，取对数后 log(alpha) = log(p(a,b|s)) - (log(M) + log(q(a)) + log(q(b)))
            log_alpha = log_p_cond_val - (np.log(NDE_intersection['proposal_M']) + log_proposal_accelerate + log_proposal_steering_angle)  
            
            # 步骤5：判断是否接受该样本，增加对 log_p_cond_val 过大情况的处理
            if log_p_cond_val > 4 or np.log(np.random.rand()) < log_alpha:  
                if type=='sample':
                    return scene_behavior, log_p_cond_val, trial_idx + 1  
                else:
                    accept+=1
                
        if type=='sample':
            print('无法采样')
        else:
            return accept/max_trials


 
