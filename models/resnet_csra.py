import importlib
import paddle
from paddle import nn
from ppcls.arch.backbone.base.theseus_layer import TheseusLayer
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url
from .csra import MHA


def build_backbone(backbone, **kwargs):
    mod = importlib.import_module('ppcls.arch.backbone.legendary_models.resnet')
    backbone = getattr(mod, backbone)(**kwargs)
    return backbone


def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


class ResNet_CSRA(TheseusLayer):
    def __init__(self, backbone, backbone_weight, num_heads, lam, class_num, input_dim=2048, pretrained=None, **kwargs):
        super(ResNet_CSRA, self).__init__()
        self.backbone = build_backbone(backbone, class_num=class_num, pretrained=backbone_weight)
        self.backbone.fc = nn.Identity()
        self.classifier = MHA(num_heads, lam, input_dim, class_num)
        if pretrained:
            _load_pretrained(pretrained, self, None, None)

    def forward(self, x):
        with paddle.static.amp.fp16_guard():
            x = self.backbone.stem(x)
            x = self.backbone.max_pool(x)
            x = self.backbone.blocks(x)
            x = self.classifier(x)
        return x
