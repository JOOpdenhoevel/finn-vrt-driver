/**
 * @file BaseDriverTest.cpp
 * @author Bjarne Wintermann (bjarne.wintermann@uni-paderborn.de) and others
 * @brief Unittest for the base driver
 * @version 0.1
 * @date 2023-10-31
 *
 * @copyright Copyright (c) 2023
 * @license All rights reserved. This program and the accompanying materials are made available under the terms of the MIT license.
 *
 */


#include <FINNCppDriver/config/FinnDriverUsedDatatypes.h>
#include <FINNCppDriver/utils/FinnUtils.h>
#include <FINNCppDriver/utils/Types.h>

#include <FINNCppDriver/core/BaseDriver.hpp>
#include <FINNCppDriver/core/DeviceBuffer/SyncDeviceBuffers.hpp>
#include <FINNCppDriver/utils/FinnDatatypes.hpp>
#include <FINNCppDriver/utils/Logger.hpp>
#include <FINNCppDriver/utils/join.hpp>

#include "gtest/gtest.h"

// Provides config and shapes
#include "UnittestConfig.h"
using namespace FinnUnittest;

class BaseDriverTest : public ::testing::Test {
     protected:
    void SetUp() override {}
    void TearDown() override {}
};


class TestDriver : public FinnUnittest::Driver<true> {
     public:
    TestDriver(const Finn::Config& pConfig, unsigned int hostBufferSize) : FinnUnittest::Driver<true>(pConfig, hostBufferSize) {}

    Finn::vector<uint8_t> inferR(const Finn::vector<uint8_t>& data, const std::string& inputBDF, const std::string& inputBufferKernelName, const std::string& outputBDF, const std::string& outputBufferKernelName, unsigned int samples) {
        return infer(data, inputBDF, inputBufferKernelName, outputBDF, outputBufferKernelName, samples);
    }

    template<typename IterType>
    Finn::vector<uint8_t> inferR(IterType first, IterType last, const std::string& inputBDF, const std::string& inputBufferKernelName, const std::string& outputBDF, const std::string& outputBufferKernelName, unsigned int samples) {
        return infer(first, last, inputBDF, inputBufferKernelName, outputBDF, outputBufferKernelName, samples);
    }
};

TEST_F(BaseDriverTest, BasicBaseDriverTest) {
    auto filler = FinnUtils::BufferFiller(0, 255);
    auto driver = TestDriver(unittestConfig, hostBufferSize);

    Finn::vector<uint8_t> data;
    Finn::vector<uint8_t> backupData;
    data.resize(driver.getTotalDataSize(bdf, inputDmaName));

    filler.fillRandom(data.begin(), data.end());
    backupData = data;

    // Setup fake output data
    driver.getDeviceHandler(bdf).getOutputBuffer(outputDmaName)->testSetMap(data);

    // Run inference
    auto results = driver.inferR(data, bdf, inputDmaName, bdf, outputDmaName, hostBufferSize);

    // Check results
    for (std::size_t i = 1; i < 9; i++) {
        EXPECT_EQ(results[0], results[i]);
    }
    EXPECT_EQ(0, results[9]);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}