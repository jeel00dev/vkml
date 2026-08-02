#include "doctest.h"

#include "vkml/util/env.h"
#include "vkml/util/assert.h"
#include "vkml/util/log.h"
#include "vkml/util/observe.h"
#include "vkml/util/decisions.h"

#include <algorithm>
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

// ---------------------------------------------------------------------------
// observe: decision facts
// ---------------------------------------------------------------------------

TEST_CASE("observe: publishing with no subscriber is inert") {
    // The property the architecture depends on: a decision site publishes and
    // learns nothing, including whether anyone is there. If this ever needed a
    // guard at the call site, the coupling this header exists to prevent would
    // have started (docs/OBSERVABILITY-ARCHITECTURE.md 4a).
    vkml::observe::subscribe(nullptr);
    CHECK_FALSE(vkml::observe::enabled());
    vkml::observe::publish({.site = "test.inert", .chose = "nothing"});
}

TEST_CASE("observe: a subscriber receives the fact it was published") {
    std::vector<std::string> seen;
    vkml::observe::subscribe([&](const vkml::observe::Decision& d) {
        seen.push_back(std::string(d.site) + ":" + std::string(d.chose) + "/" +
                       std::string(d.instead_of) + " " + std::to_string(d.required) + ">" +
                       std::to_string(d.available));
    });
    CHECK(vkml::observe::enabled());

    vkml::observe::publish({.site = "matmul.kernel",
                            .op = "matmul",
                            .chose = "gemm_naive",
                            .instead_of = "gemm_reg",
                            .because = "needs more invocations than the device allows",
                            .required = 256,
                            .available = 128});

    REQUIRE(seen.size() == 1);
    // The numbers are the point. A prose reason cannot be checked against what
    // the driver independently reports; 256 against 128 can.
    CHECK(seen[0] == "matmul.kernel:gemm_naive/gemm_reg 256>128");
    vkml::observe::subscribe(nullptr);
}

TEST_CASE("observe: a throwing subscriber cannot break the observed operation") {
    // Observation must never become a failure mode of the thing observed. A
    // matmul that fails because something was watching it is worse than no
    // observability at all.
    vkml::observe::subscribe(
        [](const vkml::observe::Decision&) { throw std::runtime_error("subscriber failure"); });
    vkml::observe::publish({.site = "test.throwing", .chose = "x"});
    vkml::observe::subscribe(nullptr);
    CHECK_FALSE(vkml::observe::enabled());
}

TEST_CASE("observe: unsubscribing leaves no live flag over a dead sink") {
    vkml::observe::subscribe([](const vkml::observe::Decision&) {});
    CHECK(vkml::observe::enabled());
    vkml::observe::subscribe(nullptr);
    CHECK_FALSE(vkml::observe::enabled());
    vkml::observe::publish({.site = "test.after", .chose = "y"});
}

TEST_CASE("decisions: the recorder is a consumer and keeps a bounded window") {
    vkml::observe::start_recording(3);
    CHECK(vkml::observe::recording());
    for (int i = 0; i < 5; ++i) {
        vkml::observe::publish({.site = "test.ring", .chose = "x", .required = i});
    }
    // Five published, three kept: the window must say so rather than let a
    // reader mistake a truncated history for a complete one.
    CHECK(vkml::observe::published() == 5);
    const auto got = vkml::observe::recorded();
    REQUIRE(got.size() == 3);
    CHECK(got.front().required == 2);  // oldest survivor
    CHECK(got.back().required == 4);   // newest
    CHECK(got.front().seq == 3);       // seq survives eviction
    CHECK(got.back().seq == 5);
    vkml::observe::stop_recording();
    CHECK_FALSE(vkml::observe::recording());
    CHECK(vkml::observe::recorded().empty());
}

TEST_CASE("decisions: recorded facts own their strings") {
    vkml::observe::start_recording(4);
    {
        // A published Decision carries string_views. If the recorder kept the
        // views rather than copying, this scope ending would dangle them.
        std::string site = "test.owned";
        std::string chose = "copied";
        vkml::observe::publish({.site = site, .chose = chose});
    }
    const auto got = vkml::observe::recorded();
    REQUIRE(got.size() == 1);
    CHECK(got[0].site == "test.owned");
    CHECK(got[0].chose == "copied");
    vkml::observe::stop_recording();
}

TEST_CASE("decisions: recording does not silence the default renderer") {
    // Two renderings of one fact are independent. An earlier draft made a
    // subscriber REPLACE the default, so installing the recorder would have
    // stopped a min-spec user being told about the kernel fallback.
    std::vector<std::string> logged;
    vkml::set_log_callback([&](vkml::LogLevel, std::string_view m) { logged.emplace_back(m); });
    vkml::observe::start_recording(4);
    vkml::observe::publish(
        {.site = "test.both", .chose = "a", .instead_of = "b", .because = "reason"});
    vkml::observe::stop_recording();
    vkml::set_log_callback(nullptr);
    CHECK(std::any_of(logged.begin(), logged.end(), [](const std::string& m) {
        return m.find("test.both") != std::string::npos;
    }));
}
