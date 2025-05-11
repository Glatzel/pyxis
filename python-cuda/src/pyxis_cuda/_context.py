class Context:
    block_size = 256


def set_block_size(size: int):
    Context.block_size = size


def get_grid_block(length: int) -> tuple[int, int]:
    grid_size = (length + Context.block_size - 1) // Context.block_size
    return grid_size, Context.block_size
