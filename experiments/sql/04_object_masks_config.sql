insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (401, 'MaskKITTI', 'iorank', 'use_masks=true', 'mask-rcnn', null, null, null, 'autoencoder',
        'use_masks=true,n_latent_features=4096,reduced_size=256',
        null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (402, 'MaskKITTI', 'iorank', 'use_masks=true', 'mask-rcnn', null, null, null, 'raw',
        'use_masks=true,reduced_size=64', null,
        null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (403, 'MaskKITTI', 'iorank', 'use_masks=true', 'mask-rcnn', null, null, null, 'alexnet',
        'use_masks=true,reduced_size=256',
        null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (404, 'MaskKITTI', 'iorank', 'use_masks=true', 'mask-rcnn', null, null, null, 'resnet',
        'use_masks=true,reduced_size=256',
        null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (405, 'MaskKITTI', 'iorank', 'use_masks=true', 'mask-rcnn', null, null, null, 'autoencoder',
        'use_masks=true,n_latent_features=4096,reduced_size=256',
        null, null,
        'torch-feta-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (406, 'MaskKITTI', 'iorank', 'use_masks=true', 'mask-rcnn', null, null, null, 'raw',
        'use_masks=true,reduced_size=64', null,
        null,
        'torch-feta-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (407, 'MaskKITTI', 'iorank', 'use_masks=true', 'mask-rcnn', null, null, null, 'alexnet',
        'use_masks=true,reduced_size=256',
        null, null,
        'torch-feta-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (408, 'MaskKITTI', 'iorank', 'use_masks=true', 'mask-rcnn', null, null, null, 'resnet',
        'use_masks=true,reduced_size=256',
        null, null,
        'torch-feta-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (409, 'MaskKITTI', 'iorank', 'use_masks=true', 'mask-rcnn', null, null, null, 'autoencoder',
        'n_latent_features=4096,reduced_size=256',
        null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (410, 'MaskKITTI', 'iorank', 'use_masks=true', 'mask-rcnn', null, null, null, 'raw', 'reduced_size=64', null,
        null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (411, 'MaskKITTI', 'iorank', 'use_masks=true', 'mask-rcnn', null, null, null, 'alexnet', 'reduced_size=256',
        null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (412, 'MaskKITTI', 'iorank', 'use_masks=true', 'mask-rcnn', null, null, null, 'resnet', 'reduced_size=256',
        null, null,
        'torch-fate-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (413, 'MaskKITTI', 'iorank', 'use_masks=true', 'mask-rcnn', null, null, null, 'autoencoder',
        'n_latent_features=4096,reduced_size=256',
        null, null,
        'torch-feta-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (414, 'MaskKITTI', 'iorank', 'use_masks=true', 'mask-rcnn', null, null, null, 'raw', 'reduced_size=64', null,
        null,
        'torch-feta-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (415, 'MaskKITTI', 'iorank', 'use_masks=true', 'mask-rcnn', null, null, null, 'alexnet', 'reduced_size=256',
        null, null,
        'torch-feta-ranker', null,
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (416, 'MaskKITTI', 'iorank', 'use_masks=true', 'mask-rcnn', null, null, null, 'resnet', 'reduced_size=256',
        null, null,
        'torch-feta-ranker', null,
        null, null, 0, 5, false);