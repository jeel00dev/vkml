#include "doctest.h"

#include "vkml/util/env.h"
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

// ---------------------------------------------------------------------------
// Environment switches
// ---------------------------------------------------------------------------

TEST_CASE("environment flags parse the way every call site used to") {
    // The rule is tested WITHOUT touching the environment. That is the point of
    // splitting parse_env_flag from env_flag: the environment is process-global
    // state, so a test that set variables would have to restore them, would
    // race any parallel test, and would still only cover the one platform's
    // setenv. Passing the string in directly removes all three problems.
    //
    // These cases are the behaviour the eighteen hand-written copies had, kept
    // deliberately rather than tightened: "0" is off and anything else is on,
    // so a switch that started rejecting "false" or "no" would silently change
    // meaning for anyone already setting them.
    CHECK(vkml::parse_env_flag("1", false));
    CHECK(vkml::parse_env_flag("anything", false));
    CHECK(vkml::parse_env_flag("false", false));  // NOT special-cased, by design
    CHECK_FALSE(vkml::parse_env_flag("0", true));

    SUBCASE("only the FIRST character decides") {
        // "0anything" is off and "10" is on, which a value parsed as an integer
        // would not agree with. Stated as its own case because it is the part of
        // the rule most likely to be "tidied" into a full parse by someone who
        // has not read this.
        CHECK_FALSE(vkml::parse_env_flag("00", true));
        CHECK_FALSE(vkml::parse_env_flag("0xyz", true));
        CHECK(vkml::parse_env_flag("10", false));
    }

    SUBCASE("unset and empty both fall back, and the fallback is honoured") {
        CHECK(vkml::parse_env_flag(nullptr, true));
        CHECK_FALSE(vkml::parse_env_flag(nullptr, false));
        CHECK(vkml::parse_env_flag("", true));
        CHECK_FALSE(vkml::parse_env_flag("", false));
    }
}

TEST_CASE("environment integers fall back only when unset or empty") {
    CHECK(vkml::parse_env_int("42", 7) == 42);
    CHECK(vkml::parse_env_int("-3", 7) == -3);
    CHECK(vkml::parse_env_int("0", 7) == 0);  // an explicit zero is not a fallback
    CHECK(vkml::parse_env_int(nullptr, 7) == 7);
    CHECK(vkml::parse_env_int("", 7) == 7);

    SUBCASE("garbage yields 0, matching the atoi these replaced") {
        // strtol returns 0 when no conversion is possible, which is what the
        // call sites' own atoi did. They clamp afterwards -- one requires more
        // than 1, another caps at 256 -- so 0 reaching them is rejected there,
        // where the valid range is visible. Pinned so that moving the clamp
        // into the helper becomes a deliberate change rather than a quiet one.
        CHECK(vkml::parse_env_int("abc", 7) == 0);
        CHECK(vkml::parse_env_int("12abc", 7) == 12);
    }

    SUBCASE("strtol rather than atoi, whose overflow is undefined") {
        // These come from a user's shell, so the value is not trusted.
        CHECK(vkml::parse_env_int("99999999999999999999", 7) != 7);
    }
}

TEST_CASE("env_value distinguishes unset from set-but-empty") {
    // getenv's own distinction, preserved because one call site tests presence
    // alone: VKML_GEMM_BLOCK being set to nothing still forces the block
    // geometry. Collapsing empty into nullopt would change that.
    //
    // A name that cannot plausibly be set is the only assertion available here
    // without mutating the environment; the empty-string half is covered by
    // parse_env_flag("") and parse_env_int("") above.
    CHECK_FALSE(vkml::env_value("VKML_A_VARIABLE_NOTHING_WOULD_EVER_SET").has_value());
    CHECK_FALSE(vkml::env_value(nullptr).has_value());
}
