/**
 * @file DeviceBufferTest.cpp
 * @author Bjarne Wintermann (bjarne.wintermann@uni-paderborn.de) and others
 * @brief Unittest for the Device Buffer
 * @version 0.1
 * @date 2023-10-31
 *
 * @copyright Copyright (c) 2023
 * @license All rights reserved. This program and the accompanying materials are made available under the terms of the MIT license.
 *
 */

#include <FINNCppDriver/core/DeviceBuffer/AsyncDeviceBuffers.hpp>
#include <FINNCppDriver/core/DeviceBuffer/SyncDeviceBuffers.hpp>
#include <FINNCppDriver/utils/FinnDatatypes.hpp>
#include <FINNCppDriver/utils/Logger.hpp>
#include <memory>
#include <random>
#include <span>

#include "gtest/gtest.h"

// VRT includes
#include "api/device.hpp"

// Provides config and shapes for testing
#include "UnittestConfig.h"
using namespace FinnUnittest;

class DBTest : public ::testing::Test {
     protected:
    vrt::Device device = vrt::Device("NULL", "../example_networks/single-layer-linear/finn_sim.vbin");

    void SetUp() override { Logger::initLogger(true); }
    void TearDown() override { device.cleanup(); }
};


TEST_F(DBTest, DBStoreTest) {
    Finn::SyncDeviceInputBuffer<uint8_t> buffer("InputBuffer", device, FinnUnittest::myShapePacked, FinnUnittest::parts);
    Finn::vector<uint8_t> data(buffer.getFeatureMapSize() * buffer.getBatchSize());
    FinnUtils::BufferFiller(0, 255).fillRandom(data.begin(), data.end());
    buffer.store({data.begin(), data.end()});
    EXPECT_EQ(buffer.testGetMap(), data);
}

TEST_F(DBTest, DBOutputTest) {
    Finn::SyncDeviceOutputBuffer<uint8_t> buffer("OutputBuffer", device, FinnUnittest::myShapePacked, FinnUnittest::parts);
    Finn::vector<uint8_t> data(buffer.getTotalDataSize());
    FinnUtils::BufferFiller(0, 255).fillRandom(data.begin(), data.end());
    buffer.testSetMap(data);
    buffer.read();
    auto vec = buffer.getData(buffer.getTotalDataSize());
    EXPECT_EQ(data, vec);
}

TEST_F(DBTest, SyncExecutionTest) {
    Finn::SyncDeviceInputBuffer<uint8_t> input_buffer("idma0", device, shape_t({1, 20, 4}), 1);
    Finn::vector<uint8_t> input_data(input_buffer.getFeatureMapSize() * input_buffer.getBatchSize());
    FinnUtils::BufferFiller(0, 255).fillRandom(input_data.begin(), input_data.end());
    EXPECT_TRUE(input_buffer.store(std::span(input_data)));
    EXPECT_TRUE(input_buffer.run());

    Finn::SyncDeviceOutputBuffer<uint8_t> output_buffer("odma0", device, shape_t({1, 20, 4}), 1);
    EXPECT_TRUE(output_buffer.run());
    EXPECT_TRUE(output_buffer.wait());
    EXPECT_TRUE(output_buffer.read());

    Finn::vector<uint8_t> output_data = output_buffer.getData(input_buffer.getFeatureMapSize() * input_buffer.getBatchSize());
    EXPECT_EQ(input_data, output_data);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}