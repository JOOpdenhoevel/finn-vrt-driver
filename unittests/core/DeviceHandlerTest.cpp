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
    std::string fn = "../example_networks/single-layer-linear/finn_sim.vbin";
    void SetUp() override {}

    void TearDown() override {}
};


TEST_F(DeviceHandlerSetup, InitTest) {
    auto devicehandler = DeviceHandler(DeviceWrapper(fn, "bb:dd:f", {std::make_shared<BufferDescriptor>("idma0", shape_t({1}))}, {std::make_shared<BufferDescriptor>("odma0", shape_t({1}))}), true, 100);

    std::vector<std::string> ionames = {"idma0", "odma0"};
    for (auto&& name : ionames) {
        vrt::Kernel kernel = devicehandler.getDevice().getKernel(name);
        ASSERT_EQ(kernel.getName(), name);
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