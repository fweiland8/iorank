-- spatial-mask (FATE)
select c.id,
       r.id,
       r.kendalls_tau,
       r.spearman,
       r.zero_one_accuracy,
       c.model_params,
       c.object_detector,
       c.feature_transformer,
       c.object_ranker,
       c.object_ranker_params
from thesis_felix.result r
         join thesis_felix.configuration c
              on r.configuration_id = c.id
where (c.id = 303 or
       c.id = 307 or
       c.id between 236 and 245)
  and r.valid = 1
  and object_ranker = 'torch-fate-ranker'
order by c.id desc, r.id desc;

-- handcrafted (FATE)
select c.id,
       r.id,
       r.kendalls_tau,
       r.spearman,
       r.zero_one_accuracy,
       c.model_params,
       c.object_detector,
       c.feature_transformer,
       c.object_ranker,
       c.object_ranker_params
from thesis_felix.result r
         join thesis_felix.configuration c
              on r.configuration_id = c.id
where (c.id = 304 or
       c.id = 308 or
       c.id between 251 and 260)
  and r.valid = 1
  and object_ranker = 'torch-fate-ranker'
order by c.id desc, r.id desc;

-- resnet (FATE)
select c.id,
       r.id,
       r.kendalls_tau,
       r.spearman,
       r.zero_one_accuracy,
       c.model_params,
       c.object_detector,
       c.feature_transformer,
       c.object_ranker,
       c.object_ranker_params
from thesis_felix.result r
         join thesis_felix.configuration c
              on r.configuration_id = c.id
where (c.id = 302 or
       c.id = 306 or
       c.id between 221 and 230
    )
  and r.valid = 1
  and kendalls_tau is not null
  and object_ranker = 'torch-fate-ranker'
order by c.id desc, r.id desc;

-- spatial-mask (FETA)
select c.id,
       r.id,
       r.kendalls_tau,
       r.spearman,
       r.zero_one_accuracy,
       c.model_params,
       c.object_detector,
       c.feature_transformer,
       c.object_ranker,
       c.object_ranker_params
from thesis_felix.result r
         join thesis_felix.configuration c
              on r.configuration_id = c.id
where (c.id = 303 or
       c.id = 307 or
       c.id between 236 and 245)
  and r.valid = 1
  and object_ranker = 'torch-feta-ranker'
order by c.id desc, r.id desc;

-- handcrafted (FETA)
select c.id,
       r.id,
       r.kendalls_tau,
       r.spearman,
       r.zero_one_accuracy,
       c.model_params,
       c.object_detector,
       c.feature_transformer,
       c.object_ranker,
       c.object_ranker_params
from thesis_felix.result r
         join thesis_felix.configuration c
              on r.configuration_id = c.id
where (c.id = 304 or
       c.id = 308 or
       c.id between 251 and 260)
  and r.valid = 1
  and object_ranker = 'torch-feta-ranker'
order by c.id desc, r.id desc;

-- resnet (FETA)
select c.id,
       r.id,
       r.kendalls_tau,
       r.spearman,
       r.zero_one_accuracy,
       c.model_params,
       c.object_detector,
       c.feature_transformer,
       c.feature_transformer_params,
       c.object_ranker,
       c.object_ranker_params
from thesis_felix.result r
         join thesis_felix.configuration c
              on r.configuration_id = c.id
where (c.id = 302 or
       c.id = 306 or
       c.id between 221 and 230
    )
  and r.valid = 1
  and object_ranker = 'torch-feta-ranker'
order by c.id desc, r.id desc;