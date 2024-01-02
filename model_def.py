import importlib
from helper.util import save_dict_to_json, reduce_tensor, adjust_learning_rate, update_dict_to_json, \
    load_pretrained_weights, load_pretrained_weights_teacher
from models import model_dict


def load_model(model_name, pretrain, n_cls, strict, gpu, multiprocessing_distributed):
    pretrained = True if pretrain == 'ImageNet' else False
    print('pretrained', pretrained)

    if  model_name == 'effiB0':
        net_def = importlib.import_module('models.efficientnet_pytorch.model7')  # dynamic import
        model = net_def.efficientnet(num_classes=n_cls, model_name='efficientnet-b0', pretrained=pretrained)

        if pretrain == 'tma_class':
            net_name = 'CLASS_ce_pretrained_ImageNet_BZ64_seed2_thread_8'
            net_idx = 250
            inf_model_path = f'/data1/trinh/data/running_result/KD/KD_PANDA_2021/class_colon_tma_manual/{net_name}/_net_{net_idx}.pth'
            print(inf_model_path)
        elif pretrain == 'PANDA':
            inf_model_path = '/data2/trinh/data/running_result/KD/CLASS_PANDA_100epoch/CLASS_ce_Effi_b0/_net_4900.pth'
            print(inf_model_path)
        elif pretrain == 'gastric_wsi':
            print('pretrained on gastric_wsi')
            inf_model_path = '/data2/trinh/data/running_result/KD/CLASS_gastric_cancer_ano0805_bright230_8class_wsi_downsample_104/CLASS_ce_EfficientNetB0_StdPre_ImageNet_BS128_CPU4_GPU2_epoch_length_50_seed2/_net_950.pth'
        elif pretrain == 'gastric_wsi_DDP':
            print('pretrained on gastric_wsi')
            inf_model_path = \
                '/data1/trinh/code/KD/SimKD_model/vanilla' \
                '/Class_ce_gastric_cancer_ano0805_bright230_8class_wsi_downsample_StdPre_ImageNet_CPU8_GPU4_epoch_length_all/' \
                'Class_ce_model_effiB0_gastric_cancer_ano0805_bright230_8class_wsi_downsample_StdPre_ImageNet_CPU8_GPU4_seed12345_epoch100/model_79.pth'
        elif pretrain == 'gastric_cancer_tma_sv0':
            print('pretrained on gastric_wsi')
            inf_model_path = \
                '/data1/trinh/code/KD/SimKD_model/vanilla' \
                '/Class_ce_gastric_cancer_tma_sv0_StdPre_ImageNet_CPU8_GPU2_epoch_length_all/' \
                'Class_ce_model_effiB0_gastric_cancer_tma_sv0_StdPre_ImageNet_CPU8_GPU2_seed12345_epoch50/model_31.pth'
        elif pretrain == 'kather19':
            inf_model_path = f'/data1/trinh/code/KD/SimKD_model/vanilla/Class_ce_kather19_StdPre_ImageNet_strict_True_CPU8_GPU2_epoch_length_all/Class_ce_model_effiB0_kather19_IS_224_BS256_StdPre_ImageNet_CPU8_GPU2_seed12345_epoch50/net_best_acc.pth'
            print(inf_model_path)
        elif pretrain == 'kather19_nonorm':
            inf_model_path = f'/data1/trinh/code/KD/SimKD_model/vanilla/Class_ce_kather19_nonorm_StdPre_ImageNet_strict_True_CPU8_GPU2_epoch_length_all/Class_ce_model_effiB0_kather19_nonorm_IS_224_BS256_StdPre_ImageNet_CPU8_GPU2_seed12345_epoch50/net_best_acc.pth'
            print(inf_model_path)
        elif pretrain == 'crc_tp_folder1_None':
            inf_model_path = f'/data1/trinh/code/KD/SimKD_model/vanilla/Class_ce_crc_tp_folder1_StdPre_None_strict_True_CPU8_GPU2_epoch_length_all/Class_ce_model_effiB0_crc_tp_folder1_IS_150_BS256_StdPre_None_CPU8_GPU2_seed12345_epoch50/net_best_acc.pth'
            print(inf_model_path)
        elif pretrain == 'crc_tp_folder1_Img':
            inf_model_path = f'/data1/trinh/code/KD/SimKD_model/vanilla/Class_ce_crc_tp_folder1_StdPre_ImageNet_strict_True_CPU8_GPU2_epoch_length_all/Class_ce_model_effiB0_crc_tp_folder1_IS_150_BS256_StdPre_ImageNet_CPU8_GPU2_seed12345_epoch50/net_best_acc.pth'
            print(inf_model_path)
        else:
            print('pretrain: ', pretrain)
            return model
        load_pretrained_weights(model, inf_model_path, gpu=gpu,
                                multiprocessing_distributed=multiprocessing_distributed, strict=strict)

    elif model_name == 'effiB2':
        net_def = importlib.import_module('models.efficientnet_pytorch.model7')  # dynamic import
        model = net_def.efficientnet(num_classes=n_cls, model_name='efficientnet-b2', pretrained=pretrained)

    elif model_name == 'ResNet50':
        model = model_dict['ResNet50'](num_classes=n_cls, pretrained=pretrained)
    elif model_name == 'resnet101':
        model = model_dict['resnet101'](num_classes=n_cls, pretrained=True)
    elif model_name == 'ResNet18':
        model = model_dict['ResNet18'](num_classes=n_cls, pretrained=pretrained)
        if pretrain == 'ssl_ciga':
            import torch
            from collections import OrderedDict
            ckpt_path = '/data1/trinh/code/KD/SimKD_model/vanilla/ozanciga_pytorchnative_ckpt_epoch_9.ckpt'
            state_dict = torch.load(ckpt_path)['state_dict']
            encoder_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'fc' not in k:
                    k = k.replace('model.resnet.', '')
                    encoder_state_dict[k] = v
            model.load_state_dict(encoder_state_dict, strict=False)

    elif model_name == 'deit_base_patch16_384':
        net_def = importlib.import_module('models.vits.vit_source_22')  # dynamic import
        model = net_def.deit_base_patch16_384(num_classes=n_cls, pretrained=pretrained)
    elif model_name == 'vit_base_patch16_384':
        net_def = importlib.import_module('models.vits.vit_source_22')  # dynamic import
        model = net_def.vit_base_patch16_384(num_classes=n_cls, pretrained=pretrained)
    elif model_name == 'deit_base_patch16_224':
        net_def = importlib.import_module('models.vits.vit_source_22')  # dynamic import
        model = net_def.deit_base_patch16_224(num_classes=n_cls, pretrained=pretrained)
    elif model_name == 'vit_base_patch16_224':
        net_def = importlib.import_module('models.vits.vit_source_22')  # dynamic import
        model = net_def.vit_base_patch16_224_in21k(num_classes=n_cls, pretrained=pretrained)

    elif model_name == 'vit_timm_base_patch16_224':
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=n_cls)
    # elif model_name == 'deit_tiny_patch16_384':
    #     net_def = importlib.import_module('models.vits.vits')  # dynamic import
    #     model = net_def.deit_tiny_patch16_224(num_classes=n_cls, pretrained=pretrained)
    elif model_name == 'vit_tiny_patch16_384':
        net_def = importlib.import_module('models.vits.vit_source_22')  # dynamic import
        model = net_def.vit_tiny_patch16_384(num_classes=n_cls, pretrained=pretrained)
    elif model_name == 'deit_tiny_patch16_224':
        net_def = importlib.import_module('models.vits.vit_source_22')  # dynamic import
        model = net_def.deit_tiny_patch16_224(num_classes=n_cls, pretrained=pretrained)
    elif model_name == 'vit_tiny_patch16_224':
        net_def = importlib.import_module('models.vits.vit_source_22')  # dynamic import
        model = net_def.vit_tiny_patch16_224_in21k(num_classes=n_cls, pretrained=pretrained)
    elif model_name == 'vit_timm_tiny_patch16_224':
        import timm
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=n_cls)
    else:
        raise NotImplementedError(model_name)
    return model



def load_model_test(model_name, pretrain, n_cls, strict, gpu, multiprocessing_distributed):
    pretrained = True if pretrain == 'ImageNet' else False
    print('pretrained', pretrained)

    net_def = importlib.import_module('models.efficientnet_pytorch.model7')  # dynamic import
    model = net_def.efficientnet(num_classes=n_cls, model_name='efficientnet-b0', pretrained=pretrained)
    model = load_pretrained_weights(model, pretrain, gpu=gpu,
                            multiprocessing_distributed=multiprocessing_distributed, strict=strict)
    return model


def load_model_test_2(model_name, pretrain, n_cls, strict):
    net_def = importlib.import_module('models.efficientnet_pytorch.model7')  # dynamic import
    pretrained = True if pretrain == 'ImageNet' else False
    print('pretrained', pretrained)
    if  pretrained == True:
        model = net_def.efficientnet(num_classes=n_cls, model_name='efficientnet-b0', pretrained=pretrained)
        return model
    else:
        model = net_def.efficientnet(num_classes=n_cls, model_name='efficientnet-b0', pretrained=pretrained)
        model = load_pretrained_weights(model, pretrain, strict=strict)
    return model


def _test():
    import torch
    net = load_model('ResNet18', 'ssl_ciga', 7, strict=False, gpu='0,1', multiprocessing_distributed=False)
    y_class = net(torch.randn(48, 3, 224, 224).cuda())
    print(y_class.size())
    # y_class = net(torch.randn(48, 3, 224, 224).cuda())
    # print(y_class.size())

    # model = net.cuda()
    # summary(model, (3, 224, 224))
if __name__ == '__main__':
    _test()
