/**
 * @file DeviceBuffer.hpp
 * @author Bjarne Wintermann (bjarne.wintermann@uni-paderborn.de), Linus Jungemann (linus.jungemann@uni-paderborn.de) and others
 * @brief Implements a wrapper to get data easier to and from the FPGAs
 * @version 2.0
 * @date 2023-12-20
 *
 * @copyright Copyright (c) 2023
 * @license All rights reserved. This program and the accompanying materials are made available under the terms of the MIT license.
 *
 */

#ifndef DEVICEBUFFER
#define DEVICEBUFFER

#include <FINNCppDriver/config/CompilationOptions.h>
#include <FINNCppDriver/utils/Types.h>

#include <FINNCppDriver/utils/FinnDatatypes.hpp>
#include <FINNCppDriver/utils/Logger.hpp>
#include <chrono>
#include <future>
#include <span>
#include <thread>

// VRT includes
#include "api/buffer.hpp"
#include "api/kernel.hpp"

namespace Finn {
    /**
     * @brief Parent class for DeviceBuffer objects.
     *
     * @tparam T Datatype in which the data is stored (e.g. uint8_t)
     */
    template<typename T>
    class DeviceBuffer {
         protected:
        /**
         * @brief Name of the buufer
         *
         */
        std::string name;
        /**
         * @brief Packed shape (Type T): (1,2,3)
         *
         */
        shapePacked_t shapePacked;
        /**
         * @brief Numbers of type T: When F has bitwidth 2, and T has bitwidth 8, the folded shape would be (1,2,10) and the packed (1,2,3) and thus 6
         *
         */
        size_t mapSize;
        /**
         * @brief VRT buffer object; This is used to interact with FPGA memory
         *
         */
        vrt::Buffer<T> internalBuffer;
        /**
         * @brief VRT kernel associated with this Buffer
         *
         */
        vrt::Kernel assocKernel;

        /**
         * @brief Total size of data in elements
         *
         */
        std::size_t totalDataSize;
        /**
         * @brief Size of feature map
         *
         */
        std::size_t featureMapSize;

         private:
         public:
        /**
         * @brief Construct a new Device Buffer object
         *
         * @param pCUName Name of compute unit
         * @param device XRT device
         * @param pAssociatedKernel XRT kernel
         * @param pShapePacked packed shape of input
         */
        DeviceBuffer(const std::string& pCUName, vrt::Device& device, const shapePacked_t& pShapePacked, unsigned int batchSize = 1)
            : name(pCUName),
              shapePacked(pShapePacked),
              mapSize(FinnUtils::getActualBufferSize(FinnUtils::shapeToElements(pShapePacked) * batchSize)),
              internalBuffer(device, mapSize, vrt::MemoryRangeType::DDR),
              assocKernel(device, pCUName) {
            shapePacked[0] = batchSize;
            FINN_LOG(loglevel::info) << "New Device Buffer of size " << mapSize * sizeof(T) << "bytes with group id " << 0 << "\n";
            FINN_LOG(loglevel::info) << "Initializing DeviceBuffer " << name << " (SHAPE PACKED: " << FinnUtils::shapeToString(pShapePacked) << " inputs of the given shape, MAP SIZE: " << mapSize << ")\n";
            for (size_t i = 0; i < mapSize; i++) {
                internalBuffer[i] = 0;
            }
            totalDataSize = FinnUtils::shapeToElements(pShapePacked) * batchSize;
            featureMapSize = totalDataSize / shapePacked[0];
            FINN_LOG(loglevel::info) << "Map has totalSize " << totalDataSize << " and featureMapSize " << featureMapSize << "\n";
        }

        /**
         * @brief Construct a new Device Buffer object (Move constructor)
         * @param buf
         */
        DeviceBuffer(DeviceBuffer&& buf) noexcept
            : name(std::move(buf.name)),
              shapePacked(std::move(buf.shapePacked)),
              mapSize(buf.mapSize),
              internalBuffer(std::move(buf.internalBuffer)),
              assocKernel(std::move(buf.assocKernel)),
              totalDataSize(buf.totalDataSize),
              featureMapSize(buf.featureMapSize) {}

        /**
         * @brief Construct a new Device Buffer object (Deleted copy constructor)
         *
         * @param buf
         */
        DeviceBuffer(const DeviceBuffer& buf) noexcept = delete;

        /**
         * @brief Prepare the DeviceBuffer for shutdown
         *
         * This function is called before the application is shutting down.
         * It can be used to release resources or perform cleanup tasks.
         * The default implementation does nothing.
         */
        virtual void prepareForShutdown() {
            // Base implementation does nothing
            FINN_LOG(loglevel::info) << "Preparing " << name << " for shutdown";
        }

        /**
         * @brief Destroy the Device Buffer object
         *
         */
        virtual ~DeviceBuffer() { FINN_LOG(loglevel::info) << "Destructing DeviceBuffer " << name << std::endl; };

        /**
         * @brief Deleted move assignment operator
         *
         * @param buf
         * @return DeviceBuffer&
         */
        DeviceBuffer& operator=(DeviceBuffer&& buf) = delete;

        /**
         * @brief Deleted copy assignment operator
         *
         * @param buf
         * @return DeviceBuffer&
         */
        DeviceBuffer& operator=(const DeviceBuffer& buf) = delete;

        /**
         * @brief Get the size in bytes of the buffer
         *
         * @return size_t Size in bytes
         */
        virtual size_t getSizeInBytes() { return totalDataSize * sizeof(T); }

        /**
         * @brief Get the feature map size
         *
         * @return size_t Feature map size
         */
        virtual size_t getFeatureMapSize() { return featureMapSize; }

        /**
         * @brief Get the batch size
         *
         * @return size_t Batch size
         */
        virtual size_t getBatchSize() { return this->shapePacked[0]; }

        /**
         * @brief Get the total data size
         *
         * @return size_t Total data size
         */
        virtual size_t getTotalDataSize() { return totalDataSize; }

        /**
         * @brief Get the name of the device buffer
         *
         * @return std::string&
         */
        virtual std::string& getName() { return name; }

        /**
         * @brief Get the Packed Shape object
         *
         * @return shape_t&
         */
        virtual shape_t& getPackedShape() { return shapePacked; }

        /**
         * @brief Run the associated kernel
         *
         * @return true Success
         * @return false Fail
         */
        virtual bool run() = 0;

        /**
         * @brief Wait for the kernel to complete execution
         *
         * @return true Success
         * @return false Failure
         */
        virtual bool wait() {
            this->assocKernel.wait();
            return true;
        };

         protected:
        /**
         * @brief Returns a device prefix for logging
         *
         * @return std::string
         */
        virtual std::string loggerPrefix() { return "[" + std::string(Finn::type_name<decltype(*this)>()) + " - " + name + "] "; }

        /**
         * @brief Synchronizes the Buffer data to the data on the FPGA
         */
        virtual void sync() = 0;

        /**
         * @brief Execute the kernel with specified repetitions
         *
         * @param repetitions Number of repetitions to execute (default: 1)
         */
        void execute(const uint32_t repetitions = 1) { assocKernel.start(this->internalBuffer.getPhysAddr(), repetitions); }
    };

    /**
     * @brief @ref SyncDeviceInputBuffer
     *
     * @tparam T
     */
    template<typename T>
    class SyncDeviceInputBuffer;
    /**
     * @brief @ref AsyncDeviceInputBuffer
     *
     * @tparam T
     */
    template<typename T>
    class AsyncDeviceInputBuffer;

    /**
     * @brief Abstract base class that defines interfaces that need to be fulfilled by the DeviceInputBuffers
     *
     * @tparam T
     */
    template<typename T>
    class DeviceInputBuffer : public DeviceBuffer<T> {
         protected:
        /**
         * @brief Specifies if DeviceBuffer is input or output buffer
         *
         */
        const IO ioMode = IO::INPUT;

         public:
        /**
         * @brief Construct a new Device Input Buffer object
         *
         * @param pName Name for indentification
         * @param device XRT device
         * @param pAssociatedKernel XRT kernel
         * @param pShapePacked packed shape of input
         */
        DeviceInputBuffer(const std::string& pCUName, vrt::Device& device, const shapePacked_t& pShapePacked, unsigned int batchSize = 1) : DeviceBuffer<T>(pCUName, device, pShapePacked, batchSize){};

        /**
         * @brief Store the given vector of data in the FPGA mem map
         * @attention This function is NOT THREAD SAFE!
         *
         * @param data
         * @return true
         * @return false
         */
        virtual bool store(std::span<const T> data) = 0;

         protected:
        /**
         * @brief Sync data from the map to the device.
         *
         */
        void sync() override { this->internalBuffer.sync(vrt::SyncType::HOST_TO_DEVICE); }

         private:
        template<typename InputIt>
        static bool storeImpl(InputIt first, InputIt last) {
            Finn::logAndError<std::runtime_error>("Base Implementation called! This should not happen.");
            return false;
        }

#ifdef UNITTEST
         public:
        Finn::vector<T> testGetMap() {
            Finn::vector<T> temp;
            for (size_t i = 0; i < FinnUtils::shapeToElements(this->shapePacked); i++) {
                temp.push_back(this->internalBuffer[i]);
            }
            return temp;
        }
        void testSyncBackFromDevice() { this->internalBuffer.sync(vrt::SyncType::DEVICE_TO_HOST); }
        vrt::Buffer<T>& testGetInternalBuffer() { return this->internalBuffer; }
#endif
    };


    /**
     * @brief Abstract base class that defines interfaces that need to be fulfilled by the DeviceOutputBuffers
     *
     * @tparam T
     */
    template<typename T>
    class DeviceOutputBuffer : public DeviceBuffer<T> {
         protected:
        /**
         * @brief Specifies IO mode of buffer
         *
         */
        const IO ioMode = IO::OUTPUT;
        /**
         * @brief Timeout for kernels
         *
         */
        unsigned int msExecuteTimeout = 1000;

         public:
        /**
         * @brief Construct a new Device Output Buffer object
         *
         * @param pName Name for indentification
         * @param device XRT device
         * @param pAssociatedKernel XRT kernel
         * @param pShapePacked packed shape of input
         */
        DeviceOutputBuffer(const std::string& pCUName, vrt::Device& device, const shapePacked_t& pShapePacked, unsigned int batchSize = 1) : DeviceBuffer<T>(pCUName, device, pShapePacked, batchSize){};

        /**
         * @brief Get the data from the buffer and return it as a vector
         *
         * @return Finn::vector<T>
         */
        virtual Finn::vector<T> getData(const std::size_t& numItems) = 0;

        /**
         * @brief Sync data from the FPGA back to the host
         *
         */
        virtual bool read() = 0;

        /**
         * @brief Register a callback that is called when data is available in the queue (Only for AsyncDeviceOutputBuffer)
         *
         * @param callback Callback function that takes the number of items available in the queue
         */
        virtual void registerCallback(std::function<void(std::size_t)> callback) {
            // Default implementation does nothing
            // This can be overridden by derived classes if needed
            Finn::logAndError<std::runtime_error>("Callback not supported by this DeviceOutputBuffer implementation.");
        }

        virtual void drain() {
            // Default implementation does nothing
            // This can be overridden by derived classes if needed
            Finn::logAndError<std::runtime_error>("Drain not supported by this DeviceOutputBuffer implementation.");
        }

         protected:
        /**
         * @brief Sync data from the FPGA into the memory map
         *
         * @return * void
         */
        void sync() override { this->internalBuffer.sync(vrt::SyncType::DEVICE_TO_HOST); }

#ifdef UNITTEST
         public:
        template<typename IteratorType>
        void testSetMap(IteratorType first, IteratorType last) {
            if (std::distance(first, last) > this->mapSize) {
                Finn::logAndError<std::length_error>("Error setting test map. Sizes dont match");
            }
            for (unsigned int i = 0; i < std::distance(first, last); ++i) {
                this->internalBuffer[i] = first[i];
            }
            this->internalBuffer.sync(vrt::SyncType::HOST_TO_DEVICE);
        }

        void testSetMap(const std::vector<T>& data) { testSetMap(data.begin(), data.end()); }

        void testSetMap(const Finn::vector<T>& data) { testSetMap(data.begin(), data.end()); }

        vrt::Buffer<T>& testGetInternalBuffer() { return this->internalBuffer; }
#endif
    };
}  // namespace Finn

#endif  // DEVICEBUFFER
