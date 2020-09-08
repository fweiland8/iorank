-- Component model results
select c.id,
       r.id,
       r.kendalls_tau,
       r.spearman,
       r.zero_one_accuracy,
       r.object_detection_precision,
       r.object_detection_recall,
       label_accuracy,
       c.object_detector,
       c.object_detector_params,
       c.feature_transformer,
       c.object_ranker
from thesis_felix.result r
         join thesis_felix.configuration c
              on r.configuration_id = c.id
where (c.id between 505 and 509 or c.id = 103)
  and valid = 1
  and kendalls_tau is not null
order by c.id desc, r.id desc;

-- E2E model results
select c.id,
       c.backbone_name,
       c.n_cells,
       c.predictions_per_cell,
       r.id,
       r.kendalls_tau,
       r.spearman,
       r.zero_one_accuracy,
       r.object_detection_precision,
       r.object_detection_recall,
       label_accuracy
from thesis_felix.e2e_result r
         join thesis_felix.e2e_configuration c
              on r.configuration_id = c.id
where c.id between 501 and 504
order by c.id desc, r.id desc;