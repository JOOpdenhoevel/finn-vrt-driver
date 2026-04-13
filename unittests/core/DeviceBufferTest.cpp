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

#include <FINNCppDriver/core/DeviceBuffer/SyncDeviceBuffers.hpp>
#include <FINNCppDriver/utils/FinnDatatypes.hpp>
#include <FINNCppDriver/utils/Logger.hpp>
#include <memory>
#include <random>
#include <span>

#include "gtest/gtest.h"

// VRT includes
#include <vrt/device.hpp>

// Provides config and shapes for testing
#include "UnittestConfig.h"
using namespace FinnUnittest;

class DBTest : public ::testing::Test {
     protected:
    std::optional<vrt::Device> device;

    void SetUp() override {
        ASSERT_STRNE(FinnUnittest::bdf.c_str(), "Change me!") << "Please set the BDF in '" << configFilePath << "' to the desired device before running the tests!";
        this->device = vrt::Device(FinnUnittest::bdf, FinnUnittest::vbinPath);
    }
    void TearDown() override {}
};

TEST_F(DBTest, DBStoreTest) {
    Finn::SyncDeviceInputBuffer<uint8_t> buffer("idma0", *device, FinnUnittest::myShapePacked, FinnUnittest::parts);
    Finn::vector<uint8_t> data(buffer.getFeatureMapSize() * buffer.getBatchSize());
    FinnUtils::BufferFiller(0, 255).fillRandom(data.begin(), data.end());
    buffer.store({data.begin(), data.end()});
    EXPECT_EQ(buffer.testGetMap(), data);
}

TEST_F(DBTest, DBOutputTest) {
    Finn::SyncDeviceOutputBuffer<uint8_t> buffer("odma0", *device, FinnUnittest::myShapePacked, FinnUnittest::parts);
    Finn::vector<uint8_t> data(buffer.getTotalDataSize());
    FinnUtils::BufferFiller(0, 255).fillRandom(data.begin(), data.end());
    buffer.testSetMap(data);
    buffer.read();
    auto vec = buffer.getData(buffer.getTotalDataSize());
    EXPECT_EQ(data, vec);
}

TEST_F(DBTest, SyncExecutionTest) {
    Finn::SyncDeviceInputBuffer<uint8_t> input_buffer("idma0", *device, FinnUnittest::myShapePacked, FinnUnittest::parts);
    Finn::vector<uint8_t> input_data(input_buffer.getFeatureMapSize() * input_buffer.getBatchSize());
    for (std::size_t i = 0; i < 20 * FinnUnittest::parts; i++) {
        reinterpret_cast<float*>(input_data.data())[i] = static_cast<float>(i);
    }
    EXPECT_TRUE(input_buffer.store(std::span(input_data)));
    EXPECT_TRUE(input_buffer.run());

    Finn::SyncDeviceOutputBuffer<uint8_t> output_buffer("odma0", *device, FinnUnittest::myShapePacked, FinnUnittest::parts);
    EXPECT_TRUE(output_buffer.run());
    EXPECT_TRUE(output_buffer.wait());
    EXPECT_TRUE(output_buffer.read());

    Finn::vector<uint8_t> output_data = output_buffer.getData(input_buffer.getFeatureMapSize() * input_buffer.getBatchSize());
    for (std::size_t i = 0; i < 20 * FinnUnittest::parts; i++) {
        float output_value = reinterpret_cast<float*>(output_data.data())[i];
        EXPECT_NEAR(output_value, std::min(1.0f, static_cast<float>(i) / 128), 0.01);
    }
}

TEST_F(DBTest, RawVRTExecutionTest) {
    vrt::Kernel idma0(*device, "idma0");
    vrt::Kernel odma0(*device, "odma0");

    vrt::Buffer<float> input_buffer(*device, 1024, idma0.portMemoryConfig("m_axi_gmem0"));
    vrt::Buffer<float> output_buffer(*device, 1024, odma0.portMemoryConfig("m_axi_gmem0"));

    for (std::size_t i = 0; i < 20 * FinnUnittest::parts; i++) {
        input_buffer[i] = static_cast<float>(i);
    }

    input_buffer.sync(vrt::SyncType::HOST_TO_DEVICE);

    idma0.start(input_buffer, FinnUnittest::parts);
    odma0.start(output_buffer, FinnUnittest::parts);

    idma0.wait();
    odma0.wait();

    output_buffer.sync(vrt::SyncType::DEVICE_TO_HOST);

    for (std::size_t i = 0; i < 20 * FinnUnittest::parts; i++) {
        EXPECT_NEAR(output_buffer[i], std::min(1.0f, static_cast<float>(i) / 128), 0.01);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}