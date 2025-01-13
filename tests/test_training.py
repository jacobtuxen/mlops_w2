from week2.train import train

def test_train():
    train(lr=1e-3, epochs=1, output_path="models/model.pt", save_model=False)
    assert True