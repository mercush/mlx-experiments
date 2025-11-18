import mlx.core as mx

world = mx.distributed.init()
x = mx.distributed.all_sum(mx.ones(world.rank()))
print(world.size(), world.rank(), x)
