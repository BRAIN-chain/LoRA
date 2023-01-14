"""Utils"""


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_buffers(model):
    return sum(p.numel() for p in model.buffers())


def summary(model):
    print(f"- # of           params:\t{count_parameters(model)}")
    print(f"- # of trainable params:\t{count_trainable_parameters(model)}")
    print(f"- # of          buffers:\t{count_buffers(model)}")
