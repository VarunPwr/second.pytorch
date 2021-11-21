from .discriminators import Discriminator

functional_networks = {
    "discriminator": Discriminator,
}


def get_network(name, cfg):
    if name not in functional_networks:
        raise ValueError("{} is not a valid network name".format(name))
    return functional_networks[name](cfg)
