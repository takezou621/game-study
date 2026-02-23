"""Tests for the dependencies utility module."""

import pytest

from utils.dependencies import (
    AIORTC_AVAILABLE,
    DependencyChecker,
    MissingDependencyError,
    NUMPY_AVAILABLE,
    OPENAI_AVAILABLE,
    PSUTIL_AVAILABLE,
    PYDANTIC_AVAILABLE,
    WEBSOCKETS_AVAILABLE,
    BaseModel,
    Field,
    deps,
    optional_import,
    require_dependency,
)


class TestOptionalImport:
    """Tests for optional_import function."""

    def test_import_existing_module(self) -> None:
        """Test importing a module that exists."""
        available, os_module = optional_import("os")
        assert available is True
        assert os_module is not None
        assert hasattr(os_module, "path")

    def test_import_nonexistent_module(self) -> None:
        """Test importing a module that doesn't exist."""
        available, module = optional_import("nonexistent_module_xyz123")
        assert available is False
        assert module is None

    def test_import_with_attribute(self) -> None:
        """Test importing a specific attribute from a module."""
        available, path_join = optional_import("os.path", attribute="join")
        assert available is True
        assert callable(path_join)

    def test_import_with_custom_package_name(self) -> None:
        """Test that package_name parameter works."""
        available, _ = optional_import("os", package_name="custom-os-package")
        assert available is True


class TestRequireDependency:
    """Tests for require_dependency function."""

    def test_require_existing_module(self) -> None:
        """Test requiring a module that exists."""
        os_module = require_dependency("os")
        assert os_module is not None
        assert hasattr(os_module, "path")

    def test_require_nonexistent_module_raises_error(self) -> None:
        """Test requiring a module that doesn't exist raises error."""
        with pytest.raises(MissingDependencyError) as exc_info:
            require_dependency("nonexistent_module_xyz123")

        assert "nonexistent_module_xyz123" in str(exc_info.value)

    def test_require_with_feature_description(self) -> None:
        """Test that feature description is included in error message."""
        with pytest.raises(MissingDependencyError) as exc_info:
            require_dependency("nonexistent_module_xyz123", feature="audio processing")

        assert "audio processing" in str(exc_info.value)

    def test_require_with_custom_package_name(self) -> None:
        """Test that custom package name is in install hint."""
        with pytest.raises(MissingDependencyError) as exc_info:
            require_dependency("nonexistent_xyz", package_name="xyz-package")

        assert "xyz-package" in str(exc_info.value)


class TestMissingDependencyError:
    """Tests for MissingDependencyError exception."""

    def test_error_message_format(self) -> None:
        """Test error message format."""
        error = MissingDependencyError("test-package")
        assert "test-package" in str(error)
        assert "pip install" in str(error)

    def test_error_with_feature(self) -> None:
        """Test error message with feature description."""
        error = MissingDependencyError("test-package", feature="audio processing")
        assert "audio processing" in str(error)
        assert "test-package" in str(error)


class TestDependencyChecker:
    """Tests for DependencyChecker class."""

    def test_check_caches_results(self) -> None:
        """Test that check results are cached."""
        checker = DependencyChecker()

        # First call
        result1 = checker.check("os")
        # Second call should return cached result
        result2 = checker.check("os")

        assert result1 is result2
        assert result1 is True

    def test_check_nonexistent_module(self) -> None:
        """Test checking a nonexistent module."""
        checker = DependencyChecker()
        result = checker.check("nonexistent_module_xyz123")
        assert result is False

    def test_has_numpy_property(self) -> None:
        """Test has_numpy property."""
        checker = DependencyChecker()
        result = checker.has_numpy
        assert isinstance(result, bool)
        # Result should be cached
        assert checker.has_numpy is result

    def test_has_openai_property(self) -> None:
        """Test has_openai property."""
        checker = DependencyChecker()
        result = checker.has_openai
        assert isinstance(result, bool)

    def test_has_pydantic_property(self) -> None:
        """Test has_pydantic property."""
        checker = DependencyChecker()
        result = checker.has_pydantic
        assert isinstance(result, bool)


class TestGlobalDepsInstance:
    """Tests for the global deps instance."""

    def test_deps_instance_exists(self) -> None:
        """Test that global deps instance exists."""
        assert deps is not None
        assert isinstance(deps, DependencyChecker)

    def test_deps_has_common_properties(self) -> None:
        """Test that deps has common property checks."""
        # These should not raise errors
        _ = deps.has_numpy
        _ = deps.has_scipy
        _ = deps.has_openai
        _ = deps.has_pydantic
        _ = deps.has_websockets
        _ = deps.has_aiortc
        _ = deps.has_psutil


class TestPreCheckedDependencies:
    """Tests for pre-checked dependency constants."""

    def test_availability_flags_are_boolean(self) -> None:
        """Test that all availability flags are boolean."""
        assert isinstance(NUMPY_AVAILABLE, bool)
        assert isinstance(OPENAI_AVAILABLE, bool)
        assert isinstance(PYDANTIC_AVAILABLE, bool)
        assert isinstance(WEBSOCKETS_AVAILABLE, bool)
        assert isinstance(AIORTC_AVAILABLE, bool)
        assert isinstance(PSUTIL_AVAILABLE, bool)


class TestPydanticFallback:
    """Tests for pydantic fallback implementation."""

    def test_basemodel_is_available(self) -> None:
        """Test that BaseModel is always available."""
        assert BaseModel is not None

    def test_field_is_available(self) -> None:
        """Test that Field is always available."""
        assert Field is not None

    def test_basemodel_can_be_instantiated(self) -> None:
        """Test that BaseModel can be instantiated."""
        model = BaseModel(test_field="value")
        assert model.test_field == "value"  # type: ignore[attr-defined]

    def test_field_returns_default(self) -> None:
        """Test that Field returns the default value."""
        result = Field(default="test_default")
        assert result == "test_default"

    def test_field_with_no_default(self) -> None:
        """Test Field with no default returns None."""
        result = Field()
        assert result is None
