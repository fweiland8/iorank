-- Experiments regarding the influence of the class label
select c.id,
       r.id,
       r.kendalls_tau,
       r.spearman,
       r.zero_one_accuracy,
       c.object_detector,
       c.feature_transformer,
       c.object_ranker
from thesis_felix.result r
         join thesis_felix.configuration c
              on r.configuration_id = c.id
where c.id between 100 and 112
  and r.valid = 1
order by c.id desc, r.id desc;

-- Experiments regarding the input type of the visual appearance feature transformers
select c.id,
       r.id,
       r.kendalls_tau,
       r.spearman,
       r.zero_one_accuracy,
       c.object_detector,
       c.feature_transformer,
       c.feature_transformer_params,
       c.object_ranker
from thesis_felix.result r
         join thesis_felix.configuration c
              on r.configuration_id = c.id
where (c.id between 101 and 104
    or c.id between 116 and 119)
  and valid = 1
order by c.id desc, r.id desc;

-- Experiments regarding combinations of feature transformers
select c.id,
       r.id,
       r.kendalls_tau,
       r.spearman,
       r.zero_one_accuracy,
       c.object_detector,
       c.feature_transformer,
       c.feature_transformer_params,
       c.object_ranker
from thesis_felix.result r
         join thesis_felix.configuration c
              on r.configuration_id = c.id
where (c.id in
       (103, 105, 106,
        113, 114, 130, 131))
  and valid = 1
  and kendalls_tau is not null
order by c.id desc, r.id desc;