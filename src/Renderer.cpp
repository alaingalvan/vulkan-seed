#include "Renderer.h"

using namespace glm;

// Vulkan Utils

void findBestExtensions(const std::vector<vk::ExtensionProperties>& installed,
                        const std::vector<const char*>& wanted,
                        std::vector<const char*>& out)
{
    for (const char* const& w : wanted)
    {
        for (vk::ExtensionProperties const& i : installed)
        {
            if (std::string(i.extensionName.data()).compare(w) == 0)
            {
                out.emplace_back(w);
                break;
            }
        }
    }
}

void findBestLayers(const std::vector<vk::LayerProperties>& installed,
                    const std::vector<const char*>& wanted,
                    std::vector<const char*>& out)
{
    for (const char* const& w : wanted)
    {
        for (vk::LayerProperties const& i : installed)
        {
            if (std::string(i.layerName.data()).compare(w) == 0)
            {
                out.emplace_back(w);
                break;
            }
        }
    }
}

uint32_t getQueueIndex(vk::PhysicalDevice& physicalDevice,
                       vk::QueueFlagBits flags)
{
    std::vector<vk::QueueFamilyProperties> queueProps =
        physicalDevice.getQueueFamilyProperties();

    for (size_t i = 0; i < queueProps.size(); ++i)
    {
        if (queueProps[i].queueFlags & flags)
        {
            return static_cast<uint32_t>(i);
        }
    }

    // Default queue index
    return 0;
}

uint32_t getMemoryTypeIndex(vk::PhysicalDevice& physicalDevice,
                            uint32_t typeBits,
                            vk::MemoryPropertyFlags properties)
{
    auto gpuMemoryProps = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < gpuMemoryProps.memoryTypeCount; i++)
    {
        if ((typeBits & 1) == 1)
        {
            if ((gpuMemoryProps.memoryTypes[i].propertyFlags & properties) ==
                properties)
            {
                return i;
            }
        }
        typeBits >>= 1;
    }
    return 0;
};

// Renderer

Renderer::Renderer(xwin::Window& window)
{
    initializeAPI(window);
    initializeResources();
    setupCommands();
    tStart = std::chrono::high_resolution_clock::now();
}

Renderer::~Renderer()
{
    mDevice.waitIdle();

    destroyCommands();
    destroyResources();
    destroyAPI();
}

void Renderer::destroyAPI()
{
    // Command Pool
    mDevice.destroyCommandPool(mCommandPool);

    // Device
    mDevice.destroy();

    // Surface
    mInstance.destroySurfaceKHR(mSurface);

    // Instance
    mInstance.destroy();
}

void Renderer::destroyFrameBuffer()
{
    // Depth Attachment
    mDevice.freeMemory(mDepthImageMemory);
    mDevice.destroyImage(mDepthImage);
    if (!mSwapchainBuffers.empty())
    {
        mDevice.destroyImageView(mSwapchainBuffers[0].views[1]);
    }

    // Image Attachments
    for (size_t i = 0; i < mSwapchainBuffers.size(); ++i)
    {
        mDevice.destroyImageView(mSwapchainBuffers[i].views[0]);
        mDevice.destroyFramebuffer(mSwapchainBuffers[i].frameBuffer);
    }
}

void Renderer::destroyCommands()
{
    mDevice.freeCommandBuffers(mCommandPool, mCommandBuffers);
}

void Renderer::initializeAPI(xwin::Window& window)
{
    /**
     * Initialize the Vulkan API by creating its various API entry points:
     */

    // 🔍 Find the best Instance Extensions

    std::vector<vk::ExtensionProperties> installedExtensions =
        vk::enumerateInstanceExtensionProperties();

    std::vector<const char*> wantedExtensions = {
        VK_KHR_SURFACE_EXTENSION_NAME,
#if defined(VK_USE_PLATFORM_WIN32_KHR)
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME
#elif defined(VK_USE_PLATFORM_MACOS_MVK)
        VK_MVK_MACOS_SURFACE_EXTENSION_NAME
#elif defined(VK_USE_PLATFORM_XCB_KHR)
        VK_KHR_XCB_SURFACE_EXTENSION_NAME
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
        VK_KHR_ANDROID_SURFACE_EXTENSION_NAME
#elif defined(VK_USE_PLATFORM_XLIB_KHR)
        VK_KHR_XLIB_SURFACE_EXTENSION_NAME
#elif defined(VK_USE_PLATFORM_XCB_KHR)
        VK_KHR_XCB_SURFACE_EXTENSION_NAME
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
        VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME
#elif defined(VK_USE_PLATFORM_MIR_KHR) || defined(VK_USE_PLATFORM_DISPLAY_KHR)
        VK_KHR_DISPLAY_EXTENSION_NAME
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
        VK_KHR_ANDROID_SURFACE_EXTENSION_NAME
#elif defined(VK_USE_PLATFORM_IOS_MVK)
        VK_MVK_IOS_SURFACE_EXTENSION_NAME
#endif
    };

    std::vector<const char*> extensions = {};

    findBestExtensions(installedExtensions, wantedExtensions, extensions);

    std::vector<const char*> wantedLayers = {
#ifdef _DEBUG
        "VK_LAYER_LUNARG_standard_validation"
#endif
    };

    std::vector<vk::LayerProperties> installedLayers =
        vk::enumerateInstanceLayerProperties();

    std::vector<const char*> layers = {};

    findBestLayers(installedLayers, wantedLayers, layers);

    // ⚪ Instance
    vk::ApplicationInfo appInfo;
    appInfo = {.pApplicationName = "MyApp",
               .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
               .pEngineName = "MyAppEngine",
               .engineVersion = VK_MAKE_VERSION(1, 0, 0),
               .apiVersion = VK_API_VERSION_1_2};

    vk::InstanceCreateInfo info(
        vk::InstanceCreateFlags(), &appInfo,
        static_cast<uint32_t>(layers.size()), layers.data(),
        static_cast<uint32_t>(extensions.size()), extensions.data());

    mInstance = vk::createInstance(info);

    // 💡 Physical Device
    std::vector<vk::PhysicalDevice> physicalDevices =
        mInstance.enumeratePhysicalDevices();
    mPhysicalDevice = physicalDevices[0];

    // 👪 Queue Family
    mQueueFamilyIndex =
        getQueueIndex(mPhysicalDevice, vk::QueueFlagBits::eGraphics);

    // ⏹ Surface
    mSurface = xgfx::getSurface(&window, mInstance);
    if (!mPhysicalDevice.getSurfaceSupportKHR(mQueueFamilyIndex, mSurface))
    {
        // Check if queueFamily supports this surface
        return;
    }

    // 📦 Queue Creation
    vk::DeviceQueueCreateInfo qcinfo;
    qcinfo.setQueueFamilyIndex(mQueueFamilyIndex);
    qcinfo.setQueueCount(1);
    mQueuePriority = 0.5f;
    qcinfo.setPQueuePriorities(&mQueuePriority);

    // 🎮 Logical Device
    std::vector<vk::ExtensionProperties> installedDeviceExtensions =
        mPhysicalDevice.enumerateDeviceExtensionProperties();

    std::vector<const char*> wantedDeviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    std::vector<const char*> deviceExtensions = {};

    findBestExtensions(installedDeviceExtensions, wantedDeviceExtensions,
                       deviceExtensions);

    vk::DeviceCreateInfo dinfo;
    dinfo.setPQueueCreateInfos(&qcinfo);
    dinfo.setQueueCreateInfoCount(1);
    dinfo.setPpEnabledExtensionNames(deviceExtensions.data());
    dinfo.setEnabledExtensionCount(
        static_cast<uint32_t>(deviceExtensions.size()));
    mDevice = mPhysicalDevice.createDevice(dinfo);

    // 📦 Queue
    mQueue = mDevice.getQueue(mQueueFamilyIndex, 0);

    // 🏊 Command Pool
    mCommandPool = mDevice.createCommandPool(vk::CommandPoolCreateInfo(
        vk::CommandPoolCreateFlags(
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer),
        mQueueFamilyIndex));

    // 🔴🟢🔵 Surface Attachment Formats
    std::vector<vk::SurfaceFormatKHR> surfaceFormats =
        mPhysicalDevice.getSurfaceFormatsKHR(mSurface);

    if (surfaceFormats.size() == 1 &&
        surfaceFormats[0].format == vk::Format::eUndefined)
        mSurfaceColorFormat = vk::Format::eB8G8R8A8Unorm;
    else
        mSurfaceColorFormat = surfaceFormats[0].format;

    mSurfaceColorSpace = surfaceFormats[0].colorSpace;

    // Since all depth formats may be optional, we need to find a suitable depth
    // format to use Start with the highest precision packed format
    std::vector<vk::Format> depthFormats = {
        vk::Format::eD32SfloatS8Uint, vk::Format::eD32Sfloat,
        vk::Format::eD24UnormS8Uint, vk::Format::eD16UnormS8Uint,
        vk::Format::eD16Unorm};

    for (vk::Format& format : depthFormats)
    {
        vk::FormatProperties depthFormatProperties =
            mPhysicalDevice.getFormatProperties(format);

        // Format must support depth stencil attachment for optimal tiling
        if (depthFormatProperties.optimalTilingFeatures &
            vk::FormatFeatureFlagBits::eDepthStencilAttachment)
        {
            mSurfaceDepthFormat = format;
            break;
        }
    }

    // Swapchain
    const xwin::WindowDesc wdesc = window.getDesc();
    setupSwapchain(wdesc.width, wdesc.height);
    mCurrentBuffer = 0;

    // Command Buffers
    createCommands();

    // Sync
    createSynchronization();
}

void Renderer::setupSwapchain(unsigned width, unsigned height)
{
    // Setup viewports, Vsync
    vk::Extent2D swapchainSize = vk::Extent2D(width, height);

    // All framebuffers / attachments will be the same size as the surface
    vk::SurfaceCapabilitiesKHR surfaceCapabilities =
        mPhysicalDevice.getSurfaceCapabilitiesKHR(mSurface);
    if (!(surfaceCapabilities.currentExtent.width == -1 ||
          surfaceCapabilities.currentExtent.height == -1))
    {
        swapchainSize = surfaceCapabilities.currentExtent;
        mRenderArea = vk::Rect2D(vk::Offset2D(), swapchainSize);
        mViewport =
            vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchainSize.width),
                         static_cast<float>(swapchainSize.height), 0, 1.0f);
    }

    // VSync
    std::vector<vk::PresentModeKHR> surfacePresentModes =
        mPhysicalDevice.getSurfacePresentModesKHR(mSurface);
    vk::PresentModeKHR presentMode = vk::PresentModeKHR::eImmediate;

    for (vk::PresentModeKHR& pm : surfacePresentModes)
    {
        if (pm == vk::PresentModeKHR::eMailbox)
        {
            presentMode = vk::PresentModeKHR::eMailbox;
            break;
        }
    }

    // Create Swapchain, Images, Frame Buffers

    mDevice.waitIdle();
    vk::SwapchainKHR oldSwapchain = mSwapchain;

    // Some devices can support more than 2 buffers, but during my tests they
    // would crash on fullscreen ~ ag Tested on an NVIDIA 1080 and 165 Hz 2K
    // display
    uint32_t backbufferCount = clamp(surfaceCapabilities.maxImageCount, 1U, 2U);

    mSwapchain = mDevice.createSwapchainKHR(vk::SwapchainCreateInfoKHR(
        vk::SwapchainCreateFlagsKHR(), mSurface, backbufferCount,
        mSurfaceColorFormat, mSurfaceColorSpace, swapchainSize, 1,
        vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive,
        1, &mQueueFamilyIndex, vk::SurfaceTransformFlagBitsKHR::eIdentity,
        vk::CompositeAlphaFlagBitsKHR::eOpaque, presentMode, VK_TRUE,
        oldSwapchain));

    mSurfaceSize = vk::Extent2D(clamp(swapchainSize.width, 1U, 8192U),
                                clamp(swapchainSize.height, 1U, 8192U));
    mRenderArea = vk::Rect2D(vk::Offset2D(), mSurfaceSize);
    mViewport = vk::Viewport(0.0f, 0.0f, static_cast<float>(mSurfaceSize.width),
                             static_cast<float>(mSurfaceSize.height), 0, 1.0f);

    // Destroy previous swapchain
    if (oldSwapchain != vk::SwapchainKHR(nullptr))
    {
        mDevice.destroySwapchainKHR(oldSwapchain);
    }

    // Resize swapchain buffers for use later
    mSwapchainBuffers.resize(backbufferCount);
}

void Renderer::initFrameBuffer()
{
    // Create Depth Image Data
    mDepthImage = mDevice.createImage(vk::ImageCreateInfo(
        vk::ImageCreateFlags(), vk::ImageType::e2D, mSurfaceDepthFormat,
        vk::Extent3D(mSurfaceSize.width, mSurfaceSize.height, 1), 1U, 1U,
        vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment |
            vk::ImageUsageFlagBits::eTransferSrc,
        vk::SharingMode::eExclusive, 1, &mQueueFamilyIndex,
        vk::ImageLayout::eUndefined));

    vk::MemoryRequirements depthMemoryReq =
        mDevice.getImageMemoryRequirements(mDepthImage);

    // Search through GPU memory properies to see if this can be device local.

    mDepthImageMemory = mDevice.allocateMemory(vk::MemoryAllocateInfo(
        depthMemoryReq.size,
        getMemoryTypeIndex(mPhysicalDevice, depthMemoryReq.memoryTypeBits,
                           vk::MemoryPropertyFlagBits::eDeviceLocal)));

    mDevice.bindImageMemory(mDepthImage, mDepthImageMemory, 0);

    vk::ImageView depthImageView =
        mDevice.createImageView(vk::ImageViewCreateInfo(
            vk::ImageViewCreateFlags(), mDepthImage, vk::ImageViewType::e2D,
            mSurfaceDepthFormat, vk::ComponentMapping(),
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth |
                                          vk::ImageAspectFlagBits::eStencil,
                                      0, 1, 0, 1)));

    std::vector<vk::Image> swapchainImages =
        mDevice.getSwapchainImagesKHR(mSwapchain);

    for (size_t i = 0; i < swapchainImages.size(); i++)
    {
        mSwapchainBuffers[i].image = swapchainImages[i];

        // Color
        mSwapchainBuffers[i].views[0] =
            mDevice.createImageView(vk::ImageViewCreateInfo(
                vk::ImageViewCreateFlags(), swapchainImages[i],
                vk::ImageViewType::e2D, mSurfaceColorFormat,
                vk::ComponentMapping(),
                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1,
                                          0, 1)));

        // Depth
        mSwapchainBuffers[i].views[1] = depthImageView;

        mSwapchainBuffers[i].frameBuffer =
            mDevice.createFramebuffer(vk::FramebufferCreateInfo(
                vk::FramebufferCreateFlags(), mRenderPass,
                static_cast<uint32_t>(mSwapchainBuffers[i].views.size()),
                mSwapchainBuffers[i].views.data(), mSurfaceSize.width,
                mSurfaceSize.height, 1));
    }
}

void Renderer::createRenderPass()
{
    std::vector<vk::AttachmentDescription> attachmentDescriptions = {
        vk::AttachmentDescription(
            vk::AttachmentDescriptionFlags(), mSurfaceColorFormat,
            vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined,
            vk::ImageLayout::ePresentSrcKHR),
        vk::AttachmentDescription(
            vk::AttachmentDescriptionFlags(), mSurfaceDepthFormat,
            vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
            vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthStencilAttachmentOptimal)};

    std::vector<vk::AttachmentReference> colorReferences = {
        vk::AttachmentReference(0, vk::ImageLayout::eColorAttachmentOptimal)};

    std::vector<vk::AttachmentReference> depthReferences = {
        vk::AttachmentReference(
            1, vk::ImageLayout::eDepthStencilAttachmentOptimal)};

    std::vector<vk::SubpassDescription> subpasses = {vk::SubpassDescription(
        vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics, 0,
        nullptr, static_cast<uint32_t>(colorReferences.size()),
        colorReferences.data(), nullptr, depthReferences.data(), 0, nullptr)};

    std::vector<vk::SubpassDependency> dependencies = {
        vk::SubpassDependency(~0U, 0, vk::PipelineStageFlagBits::eBottomOfPipe,
                              vk::PipelineStageFlagBits::eColorAttachmentOutput,
                              vk::AccessFlagBits::eMemoryRead,
                              vk::AccessFlagBits::eColorAttachmentRead |
                                  vk::AccessFlagBits::eColorAttachmentWrite,
                              vk::DependencyFlagBits::eByRegion),
        vk::SubpassDependency(0, ~0U,
                              vk::PipelineStageFlagBits::eColorAttachmentOutput,
                              vk::PipelineStageFlagBits::eBottomOfPipe,
                              vk::AccessFlagBits::eColorAttachmentRead |
                                  vk::AccessFlagBits::eColorAttachmentWrite,
                              vk::AccessFlagBits::eMemoryRead,
                              vk::DependencyFlagBits::eByRegion)};

    mRenderPass = mDevice.createRenderPass(vk::RenderPassCreateInfo(
        vk::RenderPassCreateFlags(),
        static_cast<uint32_t>(attachmentDescriptions.size()),
        attachmentDescriptions.data(), static_cast<uint32_t>(subpasses.size()),
        subpasses.data(), static_cast<uint32_t>(dependencies.size()),
        dependencies.data()));
}

void Renderer::createSynchronization()
{
    // Semaphore used to ensures that image presentation is complete before
    // starting to submit again
    mPresentCompleteSemaphore =
        mDevice.createSemaphore(vk::SemaphoreCreateInfo());

    // Semaphore used to ensures that all commands submitted have been finished
    // before submitting the image to the queue
    mRenderCompleteSemaphore =
        mDevice.createSemaphore(vk::SemaphoreCreateInfo());

    // Fence for command buffer completion
    mWaitFences.resize(mSwapchainBuffers.size());

    for (size_t i = 0; i < mWaitFences.size(); i++)
    {
        mWaitFences[i] = mDevice.createFence(
            vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
    }
}

void Renderer::initializeResources()
{
    /**
     * Create Shader uniform binding data structures:
     */

    // Descriptor Pool
    std::vector<vk::DescriptorPoolSize> descriptorPoolSizes = {
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1)};

    mDescriptorPool = mDevice.createDescriptorPool(vk::DescriptorPoolCreateInfo(
        vk::DescriptorPoolCreateFlags(), 1,
        static_cast<uint32_t>(descriptorPoolSizes.size()),
        descriptorPoolSizes.data()));

    // Descriptor Set Layout
    // Binding 0: Uniform buffer (Vertex shader)
    std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                       vk::ShaderStageFlagBits::eVertex,
                                       nullptr)};

    mDescriptorSetLayouts = {
        mDevice.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo(
            vk::DescriptorSetLayoutCreateFlags(),
            static_cast<uint32_t>(descriptorSetLayoutBindings.size()),
            descriptorSetLayoutBindings.data()))};

    mDescriptorSets =
        mDevice.allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
            mDescriptorPool,
            static_cast<uint32_t>(mDescriptorSetLayouts.size()),
            mDescriptorSetLayouts.data()));

    mPipelineLayout = mDevice.createPipelineLayout(vk::PipelineLayoutCreateInfo(
        vk::PipelineLayoutCreateFlags(),
        static_cast<uint32_t>(mDescriptorSetLayouts.size()),
        mDescriptorSetLayouts.data(), 0, nullptr));

    // Setup vertices data
    uint32_t vertexBufferSize = static_cast<uint32_t>(3) * sizeof(Vertex);

    // Setup mIndices data
    mIndices.count = 3;
    uint32_t indexBufferSize = mIndices.count * sizeof(uint32_t);

    void* data;
    // Static data like vertex and index buffer should be stored on the device
    // memory for optimal (and fastest) access by the GPU
    //
    // To achieve this we use so-called "staging buffers" :
    // - Create a buffer that's visible to the host (and can be mapped)
    // - Copy the data to this buffer
    // - Create another buffer that's local on the device (VRAM) with the same
    // size
    // - Copy the data from the host to the device using a command buffer
    // - Delete the host visible (staging) buffer
    // - Use the device local buffers for rendering

    struct StagingBuffer
    {
        vk::DeviceMemory memory;
        vk::Buffer buffer;
    };

    struct
    {
        StagingBuffer vertices;
        StagingBuffer indices;
    } stagingBuffers;

    // Vertex buffer
    stagingBuffers.vertices.buffer = mDevice.createBuffer(vk::BufferCreateInfo(
        vk::BufferCreateFlags(), vertexBufferSize,
        vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive, 1,
        &mQueueFamilyIndex));

    auto memReqs =
        mDevice.getBufferMemoryRequirements(stagingBuffers.vertices.buffer);

    // Request a host visible memory type that can be used to copy our data do
    // Also request it to be coherent, so that writes are visible to the GPU
    // right after unmapping the buffer
    stagingBuffers.vertices.memory =
        mDevice.allocateMemory(vk::MemoryAllocateInfo(
            memReqs.size,
            getMemoryTypeIndex(mPhysicalDevice, memReqs.memoryTypeBits,
                               vk::MemoryPropertyFlagBits::eHostVisible |
                                   vk::MemoryPropertyFlagBits::eHostCoherent)));

    // Map and copy
    data = mDevice.mapMemory(stagingBuffers.vertices.memory, 0, memReqs.size,
                             vk::MemoryMapFlags());
    memcpy(data, mVertexBufferData, vertexBufferSize);
    mDevice.unmapMemory(stagingBuffers.vertices.memory);
    mDevice.bindBufferMemory(stagingBuffers.vertices.buffer,
                             stagingBuffers.vertices.memory, 0);

    // Create a device local buffer to which the (host local) vertex data will
    // be copied and which will be used for rendering
    mVertices.buffer = mDevice.createBuffer(vk::BufferCreateInfo(
        vk::BufferCreateFlags(), vertexBufferSize,
        vk::BufferUsageFlagBits::eVertexBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive, 1, &mQueueFamilyIndex));

    memReqs = mDevice.getBufferMemoryRequirements(mVertices.buffer);

    mVertices.memory = mDevice.allocateMemory(vk::MemoryAllocateInfo(
        memReqs.size,
        getMemoryTypeIndex(mPhysicalDevice, memReqs.memoryTypeBits,
                           vk::MemoryPropertyFlagBits::eDeviceLocal)));

    mDevice.bindBufferMemory(mVertices.buffer, mVertices.memory, 0);

    // Index buffer
    // Copy index data to a buffer visible to the host (staging buffer)
    stagingBuffers.indices.buffer = mDevice.createBuffer(vk::BufferCreateInfo(
        vk::BufferCreateFlags(), indexBufferSize,
        vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive, 1,
        &mQueueFamilyIndex));
    memReqs =
        mDevice.getBufferMemoryRequirements(stagingBuffers.indices.buffer);
    stagingBuffers.indices.memory =
        mDevice.allocateMemory(vk::MemoryAllocateInfo(
            memReqs.size,
            getMemoryTypeIndex(mPhysicalDevice, memReqs.memoryTypeBits,
                               vk::MemoryPropertyFlagBits::eHostVisible |
                                   vk::MemoryPropertyFlagBits::eHostCoherent)));

    data = mDevice.mapMemory(stagingBuffers.indices.memory, 0, indexBufferSize,
                             vk::MemoryMapFlags());
    memcpy(data, mIndexBufferData, indexBufferSize);
    mDevice.unmapMemory(stagingBuffers.indices.memory);
    mDevice.bindBufferMemory(stagingBuffers.indices.buffer,
                             stagingBuffers.indices.memory, 0);

    // Create destination buffer with device only visibility
    mIndices.buffer = mDevice.createBuffer(
        vk::BufferCreateInfo(vk::BufferCreateFlags(), indexBufferSize,
                             vk::BufferUsageFlagBits::eTransferDst |
                                 vk::BufferUsageFlagBits::eIndexBuffer,
                             vk::SharingMode::eExclusive, 0, nullptr));

    memReqs = mDevice.getBufferMemoryRequirements(mIndices.buffer);
    mIndices.memory = mDevice.allocateMemory(vk::MemoryAllocateInfo(
        memReqs.size,
        getMemoryTypeIndex(mPhysicalDevice, memReqs.memoryTypeBits,
                           vk::MemoryPropertyFlagBits::eDeviceLocal)));

    mDevice.bindBufferMemory(mIndices.buffer, mIndices.memory, 0);

    auto getCommandBuffer = [&](bool begin) {
        vk::CommandBuffer cmdBuffer =
            mDevice.allocateCommandBuffers(vk::CommandBufferAllocateInfo(
                mCommandPool, vk::CommandBufferLevel::ePrimary, 1))[0];

        // If requested, also start the new command buffer
        if (begin)
        {
            cmdBuffer.begin(vk::CommandBufferBeginInfo());
        }

        return cmdBuffer;
    };

    // Buffer copies have to be submitted to a queue, so we need a command
    // buffer for them Note: Some devices offer a dedicated transfer queue (with
    // only the transfer bit set) that may be faster when doing lots of copies
    vk::CommandBuffer copyCmd = getCommandBuffer(true);

    // Put buffer region copies into command buffer
    std::vector<vk::BufferCopy> copyRegions = {
        vk::BufferCopy(0, 0, vertexBufferSize)};

    // Vertex buffer
    copyCmd.copyBuffer(stagingBuffers.vertices.buffer, mVertices.buffer,
                       copyRegions);

    // Index buffer
    copyRegions = {vk::BufferCopy(0, 0, indexBufferSize)};

    copyCmd.copyBuffer(stagingBuffers.indices.buffer, mIndices.buffer,
                       copyRegions);

    // Flushing the command buffer will also submit it to the queue and uses a
    // fence to ensure that all commands have been executed before returning
    auto flushCommandBuffer = [&](vk::CommandBuffer commandBuffer) {
        commandBuffer.end();

        std::vector<vk::SubmitInfo> submitInfos = {
            vk::SubmitInfo(0, nullptr, nullptr, 1, &commandBuffer, 0, nullptr)};

        // Create fence to ensure that the command buffer has finished executing
        vk::Fence fence = mDevice.createFence(vk::FenceCreateInfo());

        // Submit to the queue
        mQueue.submit(submitInfos, fence);
        // Wait for the fence to signal that command buffer has finished
        // executing
        vk::Result result;
        result = mDevice.waitForFences(1, &fence, VK_TRUE, UINT_MAX);
        mDevice.destroyFence(fence);
        mDevice.freeCommandBuffers(mCommandPool, 1, &commandBuffer);
    };

    flushCommandBuffer(copyCmd);

    // Destroy staging buffers
    // Note: Staging buffer must not be deleted before the copies have been
    // submitted and executed
    mDevice.destroyBuffer(stagingBuffers.vertices.buffer);
    mDevice.freeMemory(stagingBuffers.vertices.memory);
    mDevice.destroyBuffer(stagingBuffers.indices.buffer);
    mDevice.freeMemory(stagingBuffers.indices.memory);

    // Vertex input binding
    mVertices.inputBinding.binding = 0;
    mVertices.inputBinding.stride = sizeof(Vertex);
    mVertices.inputBinding.inputRate = vk::VertexInputRate::eVertex;

    // Inpute attribute binding describe shader attribute locations and memory
    // layouts These match the following shader layout (see
    // assets/shaders/triangle.vert):
    //	layout (location = 0) in vec3 inPos;
    //	layout (location = 1) in vec3 inColor;
    mVertices.inputAttributes.resize(2);
    // Attribute location 0: Position
    mVertices.inputAttributes[0].binding = 0;
    mVertices.inputAttributes[0].location = 0;
    mVertices.inputAttributes[0].format = vk::Format::eR32G32B32Sfloat;
    mVertices.inputAttributes[0].offset = offsetof(Vertex, position);
    // Attribute location 1: Color
    mVertices.inputAttributes[1].binding = 0;
    mVertices.inputAttributes[1].location = 1;
    mVertices.inputAttributes[1].format = vk::Format::eR32G32B32Sfloat;
    mVertices.inputAttributes[1].offset = offsetof(Vertex, color);

    // Assign to the vertex input state used for pipeline creation
    mVertices.inputState.flags = vk::PipelineVertexInputStateCreateFlags();
    mVertices.inputState.vertexBindingDescriptionCount = 1;
    mVertices.inputState.pVertexBindingDescriptions = &mVertices.inputBinding;
    mVertices.inputState.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(mVertices.inputAttributes.size());
    mVertices.inputState.pVertexAttributeDescriptions =
        mVertices.inputAttributes.data();

    // Prepare and initialize a uniform buffer block containing shader uniforms
    // Single uniforms like in OpenGL are no longer present in Vulkan. All
    // Shader uniforms are passed via uniform buffer blocks

    // Vertex shader uniform buffer block
    vk::MemoryAllocateInfo allocInfo = {};
    allocInfo.pNext = nullptr;
    allocInfo.allocationSize = 0;
    allocInfo.memoryTypeIndex = 0;

    // Create a new buffer
    mUniformDataVS.buffer = mDevice.createBuffer(
        vk::BufferCreateInfo(vk::BufferCreateFlags(), sizeof(uboVS),
                             vk::BufferUsageFlagBits::eUniformBuffer));
    // Get memory requirements including size, alignment and memory type
    memReqs = mDevice.getBufferMemoryRequirements(mUniformDataVS.buffer);
    allocInfo.allocationSize = memReqs.size;
    // Get the memory type index that supports host visible memory access
    // Most implementations offer multiple memory types and selecting the
    // correct one to allocate memory from is crucial We also want the buffer to
    // be host coherent so we don't have to flush (or sync after every update.
    // Note: This may affect performance so you might not want to do this in a
    // real world application that updates buffers on a regular base
    allocInfo.memoryTypeIndex =
        getMemoryTypeIndex(mPhysicalDevice, memReqs.memoryTypeBits,
                           vk::MemoryPropertyFlagBits::eHostVisible |
                               vk::MemoryPropertyFlagBits::eHostCoherent);
    // Allocate memory for the uniform buffer
    mUniformDataVS.memory = mDevice.allocateMemory(allocInfo);
    // Bind memory to buffer
    mDevice.bindBufferMemory(mUniformDataVS.buffer, mUniformDataVS.memory, 0);

    // Store information in the uniform's descriptor that is used by the
    // descriptor set
    mUniformDataVS.descriptor.buffer = mUniformDataVS.buffer;
    mUniformDataVS.descriptor.offset = 0;
    mUniformDataVS.descriptor.range = sizeof(uboVS);

    // Update Uniforms
    float zoom = -2.5f;

    // Update matrices
    uboVS.projectionMatrix = glm::perspective(
        45.0f, (float)mViewport.width / (float)mViewport.height, 0.01f,
        1024.0f);

    uboVS.viewMatrix =
        glm::translate(glm::identity<mat4>(), vec3(0.0f, 0.0f, zoom));

    uboVS.modelMatrix = glm::identity<mat4>();

    // Map uniform buffer and update it
    void* pData;
    pData = mDevice.mapMemory(mUniformDataVS.memory, 0, sizeof(uboVS));
    memcpy(pData, &uboVS, sizeof(uboVS));
    mDevice.unmapMemory(mUniformDataVS.memory);

    std::vector<vk::WriteDescriptorSet> descriptorWrites = {
        vk::WriteDescriptorSet(mDescriptorSets[0], 0, 0, 1,
                               vk::DescriptorType::eUniformBuffer, nullptr,
                               &mUniformDataVS.descriptor, nullptr)};

    mDevice.updateDescriptorSets(descriptorWrites, nullptr);

    // Create Render Pass

    createRenderPass();

    initFrameBuffer();

    // Create Graphics Pipeline

    std::vector<char> vertShaderCode = readFile("assets/triangle.vert.spv");
    std::vector<char> fragShaderCode = readFile("assets/triangle.frag.spv");

    mVertModule = mDevice.createShaderModule(vk::ShaderModuleCreateInfo(
        vk::ShaderModuleCreateFlags(), vertShaderCode.size(),
        (uint32_t*)vertShaderCode.data()));

    mFragModule = mDevice.createShaderModule(vk::ShaderModuleCreateInfo(
        vk::ShaderModuleCreateFlags(), fragShaderCode.size(),
        (uint32_t*)fragShaderCode.data()));

    mPipelineCache = mDevice.createPipelineCache(vk::PipelineCacheCreateInfo());

    std::vector<vk::PipelineShaderStageCreateInfo> pipelineShaderStages = {
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                          vk::ShaderStageFlagBits::eVertex,
                                          mVertModule, "main", nullptr),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                          vk::ShaderStageFlagBits::eFragment,
                                          mFragModule, "main", nullptr)};

    vk::PipelineVertexInputStateCreateInfo pvi = mVertices.inputState;

    vk::PipelineInputAssemblyStateCreateInfo pia(
        vk::PipelineInputAssemblyStateCreateFlags(),
        vk::PrimitiveTopology::eTriangleList);

    vk::PipelineViewportStateCreateInfo pv(
        vk::PipelineViewportStateCreateFlagBits(), 1, &mViewport, 1,
        &mRenderArea);

    vk::PipelineRasterizationStateCreateInfo pr(
        vk::PipelineRasterizationStateCreateFlags(), VK_FALSE, VK_FALSE,
        vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
        vk::FrontFace::eCounterClockwise, VK_FALSE, 0, 0, 0, 1.0f);

    vk::PipelineMultisampleStateCreateInfo pm(
        vk::PipelineMultisampleStateCreateFlags(), vk::SampleCountFlagBits::e1);

    // Dept and Stencil state for primative compare/test operations

    vk::PipelineDepthStencilStateCreateInfo pds =
        vk::PipelineDepthStencilStateCreateInfo(
            vk::PipelineDepthStencilStateCreateFlags(), VK_TRUE, VK_TRUE,
            vk::CompareOp::eLessOrEqual, VK_FALSE, VK_FALSE,
            vk::StencilOpState(), vk::StencilOpState(), 0, 0);

    // Blend State - How two primatives should draw on top of each other.
    std::vector<vk::PipelineColorBlendAttachmentState> colorBlendAttachments = {
        vk::PipelineColorBlendAttachmentState(
            VK_FALSE, vk::BlendFactor::eZero, vk::BlendFactor::eOne,
            vk::BlendOp::eAdd, vk::BlendFactor::eZero, vk::BlendFactor::eZero,
            vk::BlendOp::eAdd,
            vk::ColorComponentFlags(vk::ColorComponentFlagBits::eR |
                                    vk::ColorComponentFlagBits::eG |
                                    vk::ColorComponentFlagBits::eB |
                                    vk::ColorComponentFlagBits::eA))};

    vk::PipelineColorBlendStateCreateInfo pbs(
        vk::PipelineColorBlendStateCreateFlags(), 0, vk::LogicOp::eClear,
        static_cast<uint32_t>(colorBlendAttachments.size()),
        colorBlendAttachments.data());

    std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport,
                                                   vk::DynamicState::eScissor};

    vk::PipelineDynamicStateCreateInfo pdy(
        vk::PipelineDynamicStateCreateFlags(),
        static_cast<uint32_t>(dynamicStates.size()), dynamicStates.data());

    auto pipeline = mDevice.createGraphicsPipeline(
        mPipelineCache,
        vk::GraphicsPipelineCreateInfo(
            vk::PipelineCreateFlags(),
            static_cast<uint32_t>(pipelineShaderStages.size()),
            pipelineShaderStages.data(), &pvi, &pia, nullptr, &pv, &pr, &pm,
            &pds, &pbs, &pdy, mPipelineLayout, mRenderPass, 0));

    mPipeline = pipeline.value;
}

void Renderer::destroyResources()
{
    // Vertices
    mDevice.freeMemory(mVertices.memory);
    mDevice.destroyBuffer(mVertices.buffer);

    // Index buffer
    mDevice.freeMemory(mIndices.memory);
    mDevice.destroyBuffer(mIndices.buffer);

    // Shader Module
    mDevice.destroyShaderModule(mVertModule);
    mDevice.destroyShaderModule(mFragModule);

    // Render Pass
    mDevice.destroyRenderPass(mRenderPass);

    // Graphics Pipeline
    mDevice.destroyPipelineCache(mPipelineCache);
    mDevice.destroyPipeline(mPipeline);
    mDevice.destroyPipelineLayout(mPipelineLayout);

    // Descriptor Pool
    mDevice.destroyDescriptorPool(mDescriptorPool);
    for (vk::DescriptorSetLayout& dsl : mDescriptorSetLayouts)
    {
        mDevice.destroyDescriptorSetLayout(dsl);
    }

    // Uniform block object
    mDevice.freeMemory(mUniformDataVS.memory);
    mDevice.destroyBuffer(mUniformDataVS.buffer);

    // Destroy Framebuffers, Image Views
    destroyFrameBuffer();
    mDevice.destroySwapchainKHR(mSwapchain);

    // Sync
    mDevice.destroySemaphore(mPresentCompleteSemaphore);
    mDevice.destroySemaphore(mRenderCompleteSemaphore);
    for (vk::Fence& f : mWaitFences)
    {
        mDevice.destroyFence(f);
    }
}

void Renderer::createCommands()
{
    mCommandBuffers =
        mDevice.allocateCommandBuffers(vk::CommandBufferAllocateInfo(
            mCommandPool, vk::CommandBufferLevel::ePrimary,
            static_cast<uint32_t>(mSwapchainBuffers.size())));
}

void Renderer::setupCommands()
{
    std::vector<vk::ClearValue> clearValues = {
        vk::ClearColorValue(std::array<float, 4>{0.2f, 0.2f, 0.2f, 1.0f}),
        vk::ClearDepthStencilValue(1.0f, 0)};

    for (size_t i = 0; i < mCommandBuffers.size(); ++i)
    {
        vk::CommandBuffer& cmd = mCommandBuffers[i];
        cmd.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
        cmd.begin(vk::CommandBufferBeginInfo());
        cmd.beginRenderPass(
            vk::RenderPassBeginInfo(
                mRenderPass, mSwapchainBuffers[i].frameBuffer, mRenderArea,
                static_cast<uint32_t>(clearValues.size()), clearValues.data()),
            vk::SubpassContents::eInline);

        cmd.setViewport(0, 1, &mViewport);

        cmd.setScissor(0, 1, &mRenderArea);

        // Bind Descriptor Sets, these are attribute/uniform "descriptions"
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, mPipeline);

        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                               mPipelineLayout, 0, mDescriptorSets, nullptr);

        vk::DeviceSize offsets = 0;
        cmd.bindVertexBuffers(0, 1, &mVertices.buffer, &offsets);
        cmd.bindIndexBuffer(mIndices.buffer, 0, vk::IndexType::eUint32);
        cmd.drawIndexed(mIndices.count, 1, 0, 0, 1);
        cmd.endRenderPass();
        cmd.end();
    }
}

void Renderer::render()
{
    // Framelimit set to 60 fps
    tEnd = std::chrono::high_resolution_clock::now();
    float time =
        std::chrono::duration<float, std::milli>(tEnd - tStart).count();
    if (time < (1000.0f / 60.0f))
    {
        return;
    }
    tStart = std::chrono::high_resolution_clock::now();

    // Swap backbuffers
    vk::Result result;

    result = mDevice.acquireNextImageKHR(mSwapchain, UINT64_MAX,
                                         mPresentCompleteSemaphore, nullptr,
                                         &mCurrentBuffer);
    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR)
    {
        // Swapchain lost, we'll try again next poll
        resize(mSurfaceSize.width, mSurfaceSize.height);
        return;
    }
    if (result == vk::Result::eErrorDeviceLost)
    {
        // driver lost, we'll crash in this case:
        exit(1);
    }

    // Update Uniforms
    uboVS.modelMatrix =
        glm::rotate(uboVS.modelMatrix, 0.001f * time, vec3(0.0f, 1.0f, 0.0f));

    void* pData;
    pData = mDevice.mapMemory(mUniformDataVS.memory, 0, sizeof(uboVS));
    memcpy(pData, &uboVS, sizeof(uboVS));
    mDevice.unmapMemory(mUniformDataVS.memory);

    // Wait for Fences
    result = mDevice.waitForFences(1, &mWaitFences[mCurrentBuffer], VK_TRUE, UINT64_MAX);
    result = mDevice.resetFences(1, &mWaitFences[mCurrentBuffer]);

    vk::PipelineStageFlags waitDstStageMask =
        vk::PipelineStageFlagBits::eColorAttachmentOutput;
    vk::SubmitInfo submitInfo(1, &mPresentCompleteSemaphore, &waitDstStageMask,
                              1, &mCommandBuffers[mCurrentBuffer], 1,
                              &mRenderCompleteSemaphore);
    result = mQueue.submit(1, &submitInfo, mWaitFences[mCurrentBuffer]);

    if (result == vk::Result::eErrorDeviceLost)
    {
        // driver lost, we'll crash in this case:
        exit(1);
    }

    result = mQueue.presentKHR(vk::PresentInfoKHR(1, &mRenderCompleteSemaphore,
                                                  1, &mSwapchain,
                                                  &mCurrentBuffer, nullptr));

    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR)
    {
        // Swapchain lost, we'll try again next poll
        resize(mSurfaceSize.width, mSurfaceSize.height);
        return;
    }
}

void Renderer::resize(unsigned width, unsigned height)
{
    mDevice.waitIdle();
    destroyFrameBuffer();
    setupSwapchain(width, height);
    initFrameBuffer();
    destroyCommands();
    createCommands();
    setupCommands();
    mDevice.waitIdle();

    // Uniforms
    uboVS.projectionMatrix = glm::perspective(
        45.0f, (float)mViewport.width / (float)mViewport.height, 0.01f,
        1024.0f);
}
