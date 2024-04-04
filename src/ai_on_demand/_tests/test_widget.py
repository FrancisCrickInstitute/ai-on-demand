from ai_on_demand._widget import Inference
from ai_on_demand.evaluation_widget import Evaluation


def test_inference(make_napari_viewer):
    viewer = make_napari_viewer()

    inf_widget = Inference(viewer)

    assert inf_widget is not None

def test_evaluation(make_napari_viewer):
    viewer = make_napari_viewer()

    eval_widget = Evaluation(viewer)

    assert eval_widget is not None