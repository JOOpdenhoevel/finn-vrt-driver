/**
 * @file SyncInference.cpp
 * @author Linus Jungemann (linus.jungemann@uni-paderborn.de) and others
 * @brief Integrationtest for the Finn driver
 * @version 0.1
 * @date 2023-11-03
 *
 * @copyright Copyright (c) 2023
 * @license All rights reserved. This program and the accompanying materials are made available under the terms of the MIT license.
 *
 */

#include <FINNCppDriver/core/BaseDriver.hpp>
#include <FINNCppDriver/utils/FinnDatatypes.hpp>
#include <FINNCppDriver/utils/join.hpp>
#include <numeric>

#include "gtest/gtest.h"

namespace Finn {
    template<bool SynchronousInference>
    using Driver = BaseDriver<SynchronousInference, DatatypeFloat, DatatypeUInt<8>>;
}

TEST(SyncInference, syncInferenceTest) {
    std::string exampleNetworkConfig = "../example_networks/identity_net/acceleratorconfig.json";
    Finn::Config conf = Finn::createConfigFromPath(exampleNetworkConfig);
    std::string bdf = "bb:dd:f";
    std::string idma_name = conf.deviceWrappers[0].idmas[0]->kernelName;
    std::string odma_name = conf.deviceWrappers[0].odmas[0]->kernelName;

    auto driver = Finn::Driver<true>(conf, bdf, idma_name, bdf, odma_name, 1);
    std::size_t n_elements = 20;

    Finn::vector<float> data(n_elements);
    auto filler = FinnUtils::BufferFiller(0, 32);
    filler.fillRandom(data.begin(), data.end());

    // Run inference
    auto results = driver.inferSynchronous(data.begin(), data.end());

    Finn::vector<uint8_t> expectedResults(n_elements);
    for (std::size_t i = 0; i < n_elements; i++) {
        expectedResults[i] = static_cast<uint8_t>(std::round(data[i] * 2));
    }

    EXPECT_EQ(results, expectedResults);
}

TEST(SyncInference, syncBatchInferenceTest) {
    std::string exampleNetworkConfig = "../example_networks/identity_net/acceleratorconfig.json";
    Finn::Config conf = Finn::createConfigFromPath(exampleNetworkConfig);
    std::string bdf = "bb:dd:f";
    std::string idma_name = conf.deviceWrappers[0].idmas[0]->kernelName;
    std::string odma_name = conf.deviceWrappers[0].odmas[0]->kernelName;
    std::size_t n_batches = 10;
    std::size_t n_elements_per_batch = 20;
    std::size_t n_elements = n_elements_per_batch * n_batches;

    auto driver = Finn::Driver<true>(conf, bdf, idma_name, bdf, odma_name, static_cast<uint>(n_batches));

    Finn::vector<float> data(n_elements);
    auto filler = FinnUtils::BufferFiller(0, 32);
    filler.fillRandom(data.begin(), data.end());

    // Run inference
    auto results = driver.inferSynchronous(data.begin(), data.end());

    Finn::vector<uint8_t> expectedResults(n_elements);
    for (std::size_t i = 0; i < n_elements; i++) {
        expectedResults[i] = static_cast<uint8_t>(std::round(data[i] * 2));
    }

    EXPECT_EQ(results, expectedResults);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    Logger::initLogger(true);

    return RUN_ALL_TESTS();
}