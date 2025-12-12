#ifndef __SPARSE4D_BEV_COMMON_TYPES_HPP__
#define __SPARSE4D_BEV_COMMON_TYPES_HPP__

namespace sparse4d {
namespace bev {

// Output struct for NMS kernel
struct BoundingBox3D {
    float x, y, z;
    float l, w, h;
    float yaw;
    float confidence;
    int label;
    int index;
    int track_id;
};

} // namespace bev
} // namespace sparse4d

#endif // __SPARSE4D_BEV_COMMON_TYPES_HPP__

