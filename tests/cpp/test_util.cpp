#include "doctest.h"

#include "vkml/util/assert.h"
#include "vkml/util/log.h"

#include <string>
#include <vector>

TEST_CASE("VKML_CHECK throws the requested exception type") {
    CHECK_THROWS_AS(VKML_CHECK(false, ShapeError, "bad shape"), vkml::ShapeError);
    CHECK_THROWS_AS(VKML_CHECK(false, DTypeError, "bad dtype"), vkml::DTypeError);
    CHECK_THROWS_AS(VKML_CHECK(false, IndexError, "bad index"), vkml::IndexError);

    // Every vkml exception must be catchable as the common base, because the
    // binding layer relies on that to map anything unrecognised to RuntimeError.
    CHECK_THROWS_AS(VKML_CHECK(false, ShapeError, "x"), vkml::Error);

    CHECK_NOTHROW(VKML_CHECK(true, ShapeError, "not thrown"));
}

TEST_CASE("VKML_CHECK formats its message") {
    try {
        VKML_CHECK(false, ShapeError, "expected {} dims, got {}", 4, 7);
        FAIL("should have thrown");
    } catch (const vkml::ShapeError& e) {
        CHECK(std::string(e.what()) == "expected 4 dims, got 7");
    }
}

TEST_CASE("VKML_ASSERT throws InternalError and names the condition") {
    try {
        const int n = 3;
        VKML_ASSERT(n == 4, "stride table out of sync");
        FAIL("should have thrown");
    } catch (const vkml::InternalError& e) {
        const std::string what = e.what();
        CHECK(what.find("stride table out of sync") != std::string::npos);
        CHECK(what.find("n == 4") != std::string::npos);
        CHECK(what.find("test_util.cpp") != std::string::npos);
    }
}

TEST_CASE("VKML_ASSERT stays active in release builds") {
    // The whole point of VKML_ASSERT over VKML_DEBUG_ASSERT is that NDEBUG does
    // not disable it. If this ever regresses, corrupted tensors start escaping
    // into Release builds silently.
    CHECK_THROWS_AS(VKML_ASSERT(false, "always on"), vkml::InternalError);
}

TEST_CASE("VKML_NOT_IMPLEMENTED throws NotImplementedError") {
    CHECK_THROWS_AS(VKML_NOT_IMPLEMENTED("conv3d"), vkml::NotImplementedError);
}

TEST_CASE("log callback receives filtered messages") {
    std::vector<std::pair<vkml::LogLevel, std::string>> captured;

    const vkml::LogLevel saved = vkml::log_level();
    vkml::set_log_callback([&captured](vkml::LogLevel level, std::string_view msg) {
        captured.emplace_back(level, std::string(msg));
    });

    vkml::set_log_level(vkml::LogLevel::Warn);
    VKML_LOG_INFO("dropped {}", 1);
    VKML_LOG_WARN("kept {}", 2);
    VKML_LOG_ERROR("kept {}", 3);

    CHECK(captured.size() == 2);
    CHECK(captured[0].second == "kept 2");
    CHECK(captured[1].second == "kept 3");
    CHECK(captured[1].first == vkml::LogLevel::Error);

    SUBCASE("Off suppresses everything") {
        captured.clear();
        vkml::set_log_level(vkml::LogLevel::Off);
        VKML_LOG_ERROR("suppressed");
        CHECK(captured.empty());
    }

    vkml::set_log_callback(nullptr);
    vkml::set_log_level(saved);
}

TEST_CASE("log level names round-trip") {
    CHECK(vkml::to_string(vkml::LogLevel::Trace) == "trace");
    CHECK(vkml::to_string(vkml::LogLevel::Warn) == "warn");
    CHECK(vkml::to_string(vkml::LogLevel::Off) == "off");
}
