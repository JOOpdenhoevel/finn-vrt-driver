/**
 * @file Accelerator.cpp
 * @author Linus Jungemann (linus.jungemann@uni-paderborn.de), Bjarne Wintermann (bjarne.wintermann@uni-paderborn.de) and others
 * @brief Implements a wrapper to hide away details of FPGA implementation
 * @version 0.1
 * @date 2023-10-31
 *
 * @copyright Copyright (c) 2023
 * @license All rights reserved. This program and the accompanying materials are made available under the terms of the MIT license.
 *
 */

#include "Accelerator.h"

#include <FINNCppDriver/core/DeviceHandler.h>          // for DeviceHandler, UncheckedStore, ...
#include <FINNCppDriver/utils/ConfigurationStructs.h>  // IWYU pragma: keep
#include <FINNCppDriver/utils/FinnUtils.h>             // for logAndError, unreachable

#include <FINNCppDriver/utils/Logger.hpp>  // for operator<<, DevNull
#include <algorithm>                       // for count_if, find_if, tra...
#include <cstddef>                         // for size_t
#include <iterator>                        // for back_insert_iterator
#include <stdexcept>                       // for runtime_error

namespace Finn {

    Accelerator::Accelerator(const std::vector<DeviceWrapper>& deviceDefinitions, bool synchronousInference, unsigned int hostBufferSize) {
        FINN_LOG(loglevel::info) << "Constructing Accelerator\n";
        std::transform(deviceDefinitions.begin(), deviceDefinitions.end(), std::back_inserter(devices), [hostBufferSize, synchronousInference](const DeviceWrapper& dew) { return DeviceHandler(dew, synchronousInference, hostBufferSize); });
    }


    /****** GETTER / SETTER ******/
    DeviceHandler& Accelerator::getDeviceHandler(const std::string& bdf) {
        if (!containsDevice(bdf)) {
            Finn::logAndError<std::runtime_error>("Tried retrieving a deviceHandler with an unknown identifier " + bdf);
        }
        auto isCorrectHandler = [bdf](const DeviceHandler& dhh) { return dhh.getBDF() == bdf; };
        if (auto dhIt = std::find_if(devices.begin(), devices.end(), isCorrectHandler); dhIt != devices.end()) {
            return *dhIt;
        }
        FinnUtils::unreachable();
        return devices[0];
    }

    bool Accelerator::containsDevice(const std::string& bdf) {
        return std::count_if(devices.begin(), devices.end(), [bdf](const DeviceHandler& dh) { return dh.getBDF() == bdf; }) > 0;
    }

    std::vector<DeviceHandler>::iterator Accelerator::begin() { return devices.begin(); }

    std::vector<DeviceHandler>::iterator Accelerator::end() { return devices.end(); }


    /****** USER METHODS ******/

    // cppcheck-suppress unusedFunction
    [[maybe_unused]] UncheckedStore Accelerator::storeFactory(const std::string& bdf, const std::string& inputBufferKernelName) {
        if (devices.empty()) {
            Finn::logAndError<std::runtime_error>("Something went wrong. The device list should not be empty.");
        }
        if (containsDevice(bdf)) {
            DeviceHandler& devHand = getDeviceHandler(bdf);
            if (devHand.containsBuffer(inputBufferKernelName, IO::INPUT)) {
                return {devHand, inputBufferKernelName};
            }
        }
        Finn::logAndError<std::runtime_error>("Tried creating a store-closure on a deviceIndex or kernelBufferName which don't exist! Queried identifier: " + bdf + ", KernelBufferName: " + inputBufferKernelName);
        FinnUtils::unreachable();
        return {devices[0], ""};
    }

    void Accelerator::setBatchSize(uint batchsize) {
        for (auto&& elem : devices) {
            elem.setBatchSize(batchsize);
        }
    }

    bool Accelerator::run() {
        bool ret = true;
        for (auto&& dev : devices) {
            ret &= dev.run();
        }
        return ret;
    }

    bool Accelerator::wait() {
        bool ret = true;
        for (auto&& dev : devices) {
            // Each of these calls can potentielly block
            ret &= dev.wait();
        }
        return ret;
    }

    bool Accelerator::read() {
        bool ret = true;
        for (auto&& dev : devices) {
            ret &= dev.read();
        }
        return ret;
    }

    // cppcheck-suppress unusedFunction
    [[maybe_unused]] Finn::vector<uint8_t> Accelerator::getOutputData(const std::string& bdf, const std::string& outputBufferKernelName, const std::size_t& numItems) {
        if (containsDevice(bdf)) {
            FINN_LOG_DEBUG(loglevel::info) << "Retrieving results from the specified device index!";
            return getDeviceHandler(bdf).retrieveResults(outputBufferKernelName, numItems);
        } else {
            if (containsDevice(0)) {
                FINN_LOG_DEBUG(loglevel::info) << "Retrieving results from 0  device index!";
                return getDeviceHandler(0).retrieveResults(outputBufferKernelName, numItems);
            } else {
                // cppcheck-suppress missingReturn
                Finn::logAndError<std::runtime_error>("Tried receiving data in a devicehandler with an invalid deviceIndex!");
            }
        }
    }

    Finn::vector<uint8_t> Accelerator::getOutputData(const std::string& bdf, const std::string& outputBufferKernelName) {
        std::size_t numItems = getDeviceHandler(bdf).getTotalDataSize(outputBufferKernelName);
        return getDeviceHandler(bdf).retrieveResults(outputBufferKernelName, numItems);
    }

    size_t Accelerator::getSizeInBytes(const std::string& bdf, const std::string& bufferName) {
        if (containsDevice(bdf)) {
            return getDeviceHandler(bdf).getSizeInBytes(bufferName);
        }
        return 0;
    }

    size_t Accelerator::getFeatureMapSize(const std::string& bdf, const std::string& bufferName) {
        if (containsDevice(bdf)) {
            return getDeviceHandler(bdf).getFeatureMapSize(bufferName);
        }
        return 0;
    }

    size_t Accelerator::getBatchSize(const std::string& bdf, const std::string& bufferName) {
        if (containsDevice(bdf)) {
            return getDeviceHandler(bdf).getBatchSize(bufferName);
        }
        return 0;
    }

    size_t Accelerator::getTotalDataSize(const std::string& bdf, const std::string& bufferName) {
        if (containsDevice(bdf)) {
            return getDeviceHandler(bdf).getTotalDataSize(bufferName);
        }
        return 0;
    }

    void Accelerator::registerCallback(const std::string& bdf, const std::string& bufferName, std::function<void(std::size_t)> callback) {
        if (containsDevice(bdf)) {
            getDeviceHandler(bdf).registerCallback(bufferName, callback);
        } else {
            Finn::logAndError<std::runtime_error>("Tried registering a callback on a bdf which does not exist! Queried index: " + bdf + ", KernelBufferName: " + bufferName);
        }
    }

    void Accelerator::drain(const std::string& bdf, const std::string& bufferName) {
        if (containsDevice(bdf)) {
            getDeviceHandler(bdf).drain(bufferName);
        } else {
            Finn::logAndError<std::runtime_error>("Tried draining a buffer on a bdf which does not exist! Queried index: " + bdf + ", KernelBufferName: " + bufferName);
        }
    }
}  // namespace Finn
