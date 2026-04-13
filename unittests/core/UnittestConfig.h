/**
 * @file UnittestConfig.h
 * @author Bjarne Wintermann (bjarne.wintermann@uni-paderborn.de) and others
 * @brief Compile time config used in unittests
 * @version 0.1
 * @date 2023-10-31
 *
 * @copyright Copyright (c) 2023
 * @license All rights reserved. This program and the accompanying materials are made available under the terms of the MIT license.
 *
 */

#include <FINNCppDriver/utils/ConfigurationStructs.h>
#include <FINNCppDriver/utils/FinnUtils.h>
#include <FINNCppDriver/utils/Types.h>

#include <FINNCppDriver/core/BaseDriver.hpp>
#include <FINNCppDriver/utils/FinnDatatypes.hpp>
#include <FINNCppDriver/utils/Logger.hpp>
#include <array>
#include <filesystem>
#include <memory>
#include <vector>

#define MSTR(x)    #x
#define STRNGFY(x) MSTR(x)

namespace FinnUnittest {
#ifndef FINN_CUSTOM_UNITTEST_CONFIG
    const std::string configFilePath = "../example_networks/identity_net/build/driver/acceleratorconfig.json";
#else
    const std::string configFilePath = STRNGFY(FINN_CUSTOM_UNITTEST_CONFIG);
#endif

    Finn::Config unittestConfig = Finn::createConfigFromPath(std::filesystem::path(configFilePath));

    using InputFinnType = Finn::DatatypeFloat;
    using OutputFinnType = Finn::DatatypeFloat;

    template<bool SynchronousInference>
    using Driver = Finn::BaseDriver<SynchronousInference, InputFinnType, OutputFinnType>;

    Finn::DeviceWrapper device_wrapper = unittestConfig.deviceWrappers[0];
    const std::string vbinPath = device_wrapper.vbin.string();
    const std::string bdf = device_wrapper.bdf;
    const std::string inputDmaName = "idma0";
    const std::string outputDmaName = "odma0";

    auto myShapeNormal = (*std::dynamic_pointer_cast<Finn::ExtendedBufferDescriptor>(device_wrapper.idmas[0])).normalShape;
    auto myShapeFolded = (*std::dynamic_pointer_cast<Finn::ExtendedBufferDescriptor>(device_wrapper.idmas[0])).foldedShape;
    auto myShapePacked = (*std::dynamic_pointer_cast<Finn::ExtendedBufferDescriptor>(device_wrapper.idmas[0])).packedShape;

    const unsigned int hostBufferSize = 20 * 4;
    const size_t elementsPerPart = FinnUtils::shapeToElements(myShapePacked);
    const size_t parts = 10;
}  // namespace FinnUnittest