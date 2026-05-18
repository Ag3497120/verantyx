import mlx.core as mx

def test_cache():
    q = mx.random.normal((1, 40, 1, 128))
    k = mx.random.normal((1, 40, 10, 128))
    v = mx.random.normal((1, 40, 10, 128))
    
    scores = (q @ k.transpose(0, 1, 3, 2)) * (128 ** -0.5)
    print(scores.shape)
test_cache()
