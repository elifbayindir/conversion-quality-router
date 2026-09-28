import conversion_router


def test_package_importable_and_versioned():
    assert conversion_router.__version__ == "0.1.0"
