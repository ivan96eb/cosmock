import cosmock


def test_public_api_exports_generic_field_workflow():
    assert cosmock.__all__ == [
        "FieldMockFit",
        "__version__",
        "fit_field_model",
        "generate_field_mocks",
    ]

    for old_name in [
        "KappaMockFit",
        "fit_kappa_model",
        "generate_kappa_mocks",
        "fit_parameters",
        "generate_mocks",
    ]:
        assert not hasattr(cosmock, old_name)
