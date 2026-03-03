/**
 * @file DeviceHandlerTest.cpp
 * @author Linus Jungemann (linus.jungemann@uni-paderborn.de) and others
 * @brief Unittest for the device handler
 * @version 0.1
 * @date 2023-10-31
 *
 * @copyright Copyright (c) 2023
 * @license All rights reserved. This program and the accompanying materials are made available under the terms of the MIT license.
 *
 */

#include <FINNCppDriver/core/DeviceHandler.h>
#include <FINNCppDriver/utils/ConfigurationStructs.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"

using namespace Finn;

class DeviceHandlerSetup : public ::testing::Test {
     protected:
    std::string vbin_path = "../example_networks/single-layer-linear/finn_sim.vbin";
    DeviceHandler handler = DeviceHandler(DeviceWrapper(vbin_path, "bb:dd:f", {std::make_shared<BufferDescriptor>("idma0", shape_t({1, 20, 4}))}, {std::make_shared<BufferDescriptor>("odma0", shape_t({1, 20, 4}))}), true, 1);

    void SetUp() override {}

    void TearDown() override {}
};


TEST_F(DeviceHandlerSetup, InitTest) {
    std::vector<std::string> ionames = {"idma0", "odma0"};
    for (auto&& name : ionames) {
        vrt::Kernel kernel = handler.getDevice().getKernel(name);
        ASSERT_EQ(kernel.getName(), name);
    }
}

TEST_F(DeviceHandlerSetup, StoreTest) {
    std::vector<uint8_t> input_data;
    for (uint8_t i = 0; i < 1 * 20 * 4; i++) {
        input_data.push_back(i);
    }
    handler.store(std::span(input_data), "idma0");

    Finn::vector<uint8_t> input_buffer = handler.getInputBuffer("idma0")->testGetMap();
    ASSERT_EQ(input_buffer.size(), 20 * 4);
    for (uint8_t i = 0; i < 1 * 20 * 4; i++) {
        ASSERT_EQ(input_buffer[i], i);
    }
}

TEST_F(DeviceHandlerSetup, RunTest) {
    std::vector<uint8_t> input_data;
    for (uint8_t i = 0; i < 1 * 20 * 4; i++) {
        input_data.push_back(i);
    }
    handler.store(std::span(input_data), "idma0");

    handler.run();
    handler.wait();
    ASSERT_TRUE(handler.read());

    Finn::vector<uint8_t> results = handler.retrieveResults("odma0", 20 * 4);
    ASSERT_EQ(results.size(), 20 * 4);
    for (uint8_t i = 0; i < 1 * 20 * 4; i++) {
        ASSERT_EQ(results[i], i);
    }
}

/*
TODO: Reenable once exception handling in VRT is more stable
TEST_F(DeviceHandlerSetup, ArgumentTest) {
    EXPECT_THROW(DeviceHandler(DeviceWrapper("", "bb:dd:f", {std::make_shared<BufferDescriptor>("idma0", shape_t({1}))}, {std::make_shared<BufferDescriptor>("odma0", shape_t({1}))}), true, 1), std::filesystem::filesystem_error);
    EXPECT_THROW(DeviceHandler(DeviceWrapper(fn, "bb:dd:f", {}, {std::make_shared<BufferDescriptor>("odma0", shape_t({1}))}), true, 1), std::invalid_argument);
    EXPECT_THROW(DeviceHandler(DeviceWrapper(fn, "bb:dd:f", {std::make_shared<BufferDescriptor>("", shape_t({1}))}, {std::make_shared<BufferDescriptor>("odma0", shape_t({1}))}), true, 1), std::invalid_argument);
    EXPECT_THROW(DeviceHandler(DeviceWrapper(fn, "bb:dd:f", {std::make_shared<BufferDescriptor>("idma0", shape_t({}))}, {std::make_shared<BufferDescriptor>("odma0", shape_t({1}))}), true, 1), std::invalid_argument);
    EXPECT_THROW(DeviceHandler(DeviceWrapper(fn, "bb:dd:f", {std::make_shared<BufferDescriptor>("idma0", shape_t({1}))}, {std::make_shared<BufferDescriptor>("", shape_t({1}))}), true, 1), std::invalid_argument);
    EXPECT_THROW(DeviceHandler(DeviceWrapper(fn, "bb:dd:f", {std::make_shared<BufferDescriptor>("idma0", shape_t({1}))}, {std::make_shared<BufferDescriptor>("odma0", shape_t({}))}), true, 1), std::invalid_argument);
    EXPECT_THROW(DeviceHandler(DeviceWrapper(fn, "bb:dd:f", {std::make_shared<BufferDescriptor>("idma0", shape_t({1}))}, {}), true, 1), std::invalid_argument);
    EXPECT_NO_THROW(DeviceHandler(DeviceWrapper(fn, "bb:dd:f", {std::make_shared<BufferDescriptor>("idma0", shape_t({1})), std::make_shared<BufferDescriptor>("c", shape_t({1}))}, {std::make_shared<BufferDescriptor>("odma0", shape_t({1,
2}))}), true, 1));
}
*/

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}