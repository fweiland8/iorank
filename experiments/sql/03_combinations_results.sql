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
where c.id between 301 and 399
  and r.valid = 1
order by c.id desc, r.id desc;