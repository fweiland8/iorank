insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (221, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'resnet', 'reduced_size=128', null,
        null,
        'torch-fate-ranker', 'use_image_context=true',
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (222, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'resnet', 'reduced_size=128', null,
        null,
        'torch-feta-ranker', 'use_image_context=true',
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (223, 'KITTI', 'iorank', 'box_expansion_factor=0.3', 'faster-rcnn', null, null, null,
        'resnet', 'reduced_size=128', null,
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
values (224, 'KITTI', 'iorank', 'box_expansion_factor=0.3', 'faster-rcnn', null, null, null,
        'resnet', 'reduced_size=128', null,
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
values (225, 'KITTI', 'iorank', 'box_expansion_factor=1.0', 'faster-rcnn', null, null, null,
        'resnet', 'reduced_size=128', null,
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
values (226, 'KITTI', 'iorank', 'box_expansion_factor=1.0', 'faster-rcnn', null, null, null,
        'resnet', 'reduced_size=128', null,
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
values (227, 'KITTI', 'iorank', 'box_expansion_factor=0.3', 'faster-rcnn', null, null, null,
        'resnet', 'reduced_size=128', null,
        null,
        'torch-fate-ranker', 'use_image_context=true',
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (228, 'KITTI', 'iorank', 'box_expansion_factor=0.3', 'faster-rcnn', null, null, null,
        'resnet', 'reduced_size=128', null,
        null,
        'torch-feta-ranker', 'use_image_context=true',
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (229, 'KITTI', 'iorank', 'box_expansion_factor=1.0', 'faster-rcnn', null, null, null,
        'resnet', 'reduced_size=128', null,
        null,
        'torch-fate-ranker', 'use_image_context=true',
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (230, 'KITTI', 'iorank', 'box_expansion_factor=1.0', 'faster-rcnn', null, null, null,
        'resnet', 'reduced_size=128', null,
        null,
        'torch-feta-ranker', 'use_image_context=true',
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (236, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'spatial-mask', null, null,
        null,
        'torch-fate-ranker', 'use_image_context=true',
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (237, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'spatial-mask', null, null,
        null,
        'torch-feta-ranker', 'use_image_context=true',
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (238, 'KITTI', 'iorank', 'box_expansion_factor=0.3', 'faster-rcnn', null, null, null,
        'spatial-mask', null, null,
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
values (239, 'KITTI', 'iorank', 'box_expansion_factor=0.3', 'faster-rcnn', null, null, null,
        'spatial-mask', null, null,
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
values (240, 'KITTI', 'iorank', 'box_expansion_factor=1.0', 'faster-rcnn', null, null, null,
        'spatial-mask', null, null,
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
values (241, 'KITTI', 'iorank', 'box_expansion_factor=1.0', 'faster-rcnn', null, null, null,
        'spatial-mask', null, null,
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
values (242, 'KITTI', 'iorank', 'box_expansion_factor=0.3', 'faster-rcnn', null, null, null,
        'spatial-mask', null, null,
        null,
        'torch-fate-ranker', 'use_image_context=true',
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (243, 'KITTI', 'iorank', 'box_expansion_factor=0.3', 'faster-rcnn', null, null, null,
        'spatial-mask', null, null,
        null,
        'torch-feta-ranker', 'use_image_context=true',
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (244, 'KITTI', 'iorank', 'box_expansion_factor=1.0', 'faster-rcnn', null, null, null,
        'spatial-mask', null, null,
        null,
        'torch-fate-ranker', 'use_image_context=true',
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (245, 'KITTI', 'iorank', 'box_expansion_factor=1.0', 'faster-rcnn', null, null, null,
        'spatial-mask', null, null,
        null,
        'torch-feta-ranker', 'use_image_context=true',
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (251, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'handcrafted', null, null,
        null,
        'torch-fate-ranker', 'use_image_context=true',
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (252, 'KITTI', 'iorank', null, 'faster-rcnn', null, null, null, 'handcrafted', null, null,
        null,
        'torch-feta-ranker', 'use_image_context=true',
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (253, 'KITTI', 'iorank', 'box_expansion_factor=0.3', 'faster-rcnn', null, null, null,
        'handcrafted', null, null,
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
values (254, 'KITTI', 'iorank', 'box_expansion_factor=0.3', 'faster-rcnn', null, null, null,
        'handcrafted', null, null,
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
values (255, 'KITTI', 'iorank', 'box_expansion_factor=1.0', 'faster-rcnn', null, null, null,
        'handcrafted', null, null,
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
values (256, 'KITTI', 'iorank', 'box_expansion_factor=1.0', 'faster-rcnn', null, null, null,
        'handcrafted', null, null,
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
values (257, 'KITTI', 'iorank', 'box_expansion_factor=0.3', 'faster-rcnn', null, null, null,
        'handcrafted', null, null,
        null,
        'torch-fate-ranker', 'use_image_context=true',
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (258, 'KITTI', 'iorank', 'box_expansion_factor=0.3', 'faster-rcnn', null, null, null,
        'handcrafted', null, null,
        null,
        'torch-feta-ranker', 'use_image_context=true',
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (259, 'KITTI', 'iorank', 'box_expansion_factor=1.0', 'faster-rcnn', null, null, null,
        'handcrafted', null, null,
        null,
        'torch-fate-ranker', 'use_image_context=true',
        null, null, 0, 5, false);

insert into thesis_felix.configuration(id, dataset, model, model_params, object_detector, object_detector_params,
                                       object_detector_param_ranges, object_detector_trainer_param_ranges,
                                       feature_transformer, feature_transformer_params,
                                       feature_transformer_param_ranges, feature_transformer_trainer_param_ranges,
                                       object_ranker, object_ranker_params, object_ranker_param_ranges,
                                       object_ranker_trainer_param_ranges, tuning_iterations, n_experiments,
                                       augmentation)
values (260, 'KITTI', 'iorank', 'box_expansion_factor=1.0', 'faster-rcnn', null, null, null,
        'handcrafted', null, null,
        null,
        'torch-feta-ranker', 'use_image_context=true',
        null, null, 0, 5, false);