def epoch_time(t0, t1):
    elapsed_time = t1-t0
    HH = int(elapsed_time/3600)
    MM = int((elapsed_time-HH*3600)/60)
    SS = elapsed_time - HH*3600 - MM*60
    return "Time: {:d}h {:d}m {:0.1f}s".format(HH, MM, SS)


def initialize_log(args, type='classification'):
    img_size=(args.img_size,args.img_size)

    log_dict = dict()
    log_dict["dataset"] = args.dataset
    log_dict["training_images"] = 0
    log_dict["validation_images"] = 0
    log_dict["DA_Library"] = args.da_library
    log_dict["DA_Level"] = args.da_level
    log_dict["model_name"] = args.model_name
    log_dict["backbone"] = args.backbone
    log_dict["model_type"] = type
    if type=='siamese': log_dict["margin"] = args.margin
    log_dict["weights"] = args.weights
    log_dict["image_size"] = img_size
    log_dict["classes"] = args.classes
    if type=='siamese': log_dict["embedding_size"] = args.embedding_size
    log_dict["epochs"] = args.epochs
    log_dict["bs"] = args.batch_size
    log_dict["lr"] = args.learning_rate
    log_dict["lr_update_freq"] = args.lr_update_freq
    log_dict["jobs"] = args.jobs
    if type=='siamese': log_dict["number_of_templates"] = args.templates

    return log_dict
