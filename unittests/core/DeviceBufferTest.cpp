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
    vrt::Device device = vrt::Device("0000:21:00", "../example_networks/identity_net/build/bitfile/finn-accel.vbin");

    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(DBTest, DBStoreTest) {
    Finn::SyncDeviceInputBuffer<uint8_t> buffer("idma0", device, FinnUnittest::myShapePacked, FinnUnittest::parts);
    Finn::vector<uint8_t> data(buffer.getFeatureMapSize() * buffer.getBatchSize());
    FinnUtils::BufferFiller(0, 255).fillRandom(data.begin(), data.end());
    buffer.store({data.begin(), data.end()});
    EXPECT_EQ(buffer.testGetMap(), data);
}

TEST_F(DBTest, DBOutputTest) {
    Finn::SyncDeviceOutputBuffer<uint8_t> buffer("odma0", device, FinnUnittest::myShapePacked, FinnUnittest::parts);
    Finn::vector<uint8_t> data(buffer.getTotalDataSize());
    FinnUtils::BufferFiller(0, 255).fillRandom(data.begin(), data.end());
    buffer.testSetMap(data);
    buffer.read();
    auto vec = buffer.getData(buffer.getTotalDataSize());
    EXPECT_EQ(data, vec);
}

TEST_F(DBTest, SyncExecutionTest) {
    Finn::SyncDeviceInputBuffer<uint8_t> input_buffer("idma0", device, shape_t({1, 10, 1}), 1);
    Finn::vector<uint8_t> input_data(input_buffer.getFeatureMapSize() * input_buffer.getBatchSize());
    FinnUtils::BufferFiller(0, 255).fillRandom(input_data.begin(), input_data.end());
    EXPECT_TRUE(input_buffer.store(std::span(input_data)));
    EXPECT_TRUE(input_buffer.run());

    Finn::SyncDeviceOutputBuffer<uint8_t> output_buffer("odma0", device, shape_t({1, 10, 1}), 1);
    EXPECT_TRUE(output_buffer.run());
    EXPECT_TRUE(output_buffer.wait());
    EXPECT_TRUE(output_buffer.read());

    Finn::vector<uint8_t> output_data = output_buffer.getData(input_buffer.getFeatureMapSize() * input_buffer.getBatchSize());
    for (std::size_t i = 1; i < 9; i++) {
        EXPECT_EQ(output_data[0], output_data[i]);
    }
    EXPECT_EQ(0, output_data[9]);
}

TEST_F(DBTest, RawVRTExecutionTest) {
    vrt::Kernel idma0(device, "idma0");
    vrt::Kernel odma0(device, "odma0");

    vrt::Buffer<float> input_buffer(device, 1024, idma0.portMemoryConfig("m_axi_gmem0"));
    vrt::Buffer<float> output_buffer(device, 1024, odma0.portMemoryConfig("m_axi_gmem0"));
    std::cout << "Allocated buffers" << std::endl;

    for (std::size_t i = 0; i < 20; i++) {
        input_buffer[i] = static_cast<float>(i);
    }

    input_buffer.sync(vrt::SyncType::HOST_TO_DEVICE);
    std::cout << "Synched data to the device" << std::endl;

    idma0.start(input_buffer, static_cast<uint32_t>(1));
    odma0.start(output_buffer, static_cast<uint32_t>(1));
    std::cout << "Started the kernels" << std::endl;

    idma0.wait();
    std::cout << "IDMA done" << std::endl;
    odma0.wait();
    std::cout << "ODMA done" << std::endl;

    output_buffer.sync(vrt::SyncType::DEVICE_TO_HOST);
    std::cout << "Results synched from the device" << std::endl;

    for (std::size_t i = 0; i < 20; i++) {
        EXPECT_EQ(output_buffer[i], static_cast<float>(i) * 0.00784313772f);
    }
    std::cout << "Outputs verified, done!" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}