-- Table containing the experiment configuration for the component model
create table thesis_felix.configuration
(
    id                                       serial not null,
    dataset                                  varchar,
    model                                    varchar,
    model_params                             varchar,
    object_detector                          varchar,
    object_detector_params                   varchar,
    object_detector_param_ranges             varchar,
    object_detector_trainer_param_ranges     varchar,
    feature_transformer                      varchar,
    feature_transformer_params               varchar,
    feature_transformer_param_ranges         varchar,
    feature_transformer_trainer_param_ranges varchar,
    object_ranker                            varchar,
    object_ranker_params                     varchar,
    object_ranker_param_ranges               varchar,
    object_ranker_trainer_param_ranges       varchar,
    tuning_iterations                        integer,
    n_experiments                            integer,
    augmentation                             boolean
);

-- Table containing the experiment configuration for the E2E model
create table thesis_felix.e2e_configuration
(
    id                   serial not null,
    dataset              varchar,
    n_cells              integer,
    predictions_per_cell integer,
    input_size           integer,
    backbone_name        varchar,
    max_n_epochs         integer,
    lr_decay_epoch       integer,
    od_loss_factor       real,
    or_loss_factor       real,
    batch_size           integer,
    n_experiments        integer,
    include_spatial_mask boolean,
    feature_reduction    varchar
);

-- Table containing the experiment results for the component model
create table thesis_felix.result
(
    id                         serial  not null,
    configuration_id           integer not null,
    kendalls_tau               real,
    spearman                   real,
    zero_one_accuracy          real,
    exception                  varchar,
    started                    timestamp default now(),
    finished                   timestamp,
    duration                   integer,
    hostname                   varchar,
    object_detection_precision real,
    object_detection_recall    real,
    average_ranking_size       real,
    label_accuracy             real,
    valid                      integer   default 1
);

-- Table containing the experiment results for the E2E model
create table thesis_felix.e2e_result
(
    id                         serial not null,
    configuration_id           integer,
    kendalls_tau               real,
    spearman                   real,
    zero_one_accuracy          real,
    exception                  varchar,
    started                    timestamp default now(),
    finished                   timestamp,
    duration                   integer,
    hostname                   varchar,
    object_detection_precision real,
    object_detection_recall    real,
    average_ranking_size       real,
    label_accuracy             real
);






