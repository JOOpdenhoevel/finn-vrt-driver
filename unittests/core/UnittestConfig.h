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
    const std::string configFilePath = "../example_networks/single-layer-linear/acceleratorconfig.json";
    const std::string vbinPath = "../example_networks/single-layer-linear/finn_sim.vbin";
#else
    const std::string configFilePath = STRNGFY(FINN_CUSTOM_UNITTEST_CONFIG);
#endif

    Finn::Config unittestConfig = Finn::createConfigFromPath(std::filesystem::path(configFilePath));

    using InputFinnType = Finn::DatatypeFloat;
    using OutputFinnType = Finn::DatatypeInt<5>;

    template<bool SynchronousInference>
    using Driver = Finn::BaseDriver<SynchronousInference, InputFinnType, OutputFinnType>;

    const std::string bdf = "bb:dd:f";
    const std::string inputDmaName = "idma0";
    const std::string outputDmaName = "odma0";

    auto myShapeNormal = (*std::dynamic_pointer_cast<Finn::ExtendedBufferDescriptor>(unittestConfig.deviceWrappers[0].idmas[0])).normalShape;
    auto myShapeFolded = (*std::dynamic_pointer_cast<Finn::ExtendedBufferDescriptor>(unittestConfig.deviceWrappers[0].idmas[0])).foldedShape;
    auto myShapePacked = (*std::dynamic_pointer_cast<Finn::ExtendedBufferDescriptor>(unittestConfig.deviceWrappers[0].idmas[0])).packedShape;

    const unsigned int hostBufferSize = 10;
    const size_t elementsPerPart = FinnUtils::shapeToElements(myShapePacked);
    const size_t parts = 10;
}  // namespace FinnUnittest