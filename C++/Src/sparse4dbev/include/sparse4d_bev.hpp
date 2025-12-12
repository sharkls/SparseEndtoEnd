#ifndef __SPARSE4D_BEV_HPP__
#define __SPARSE4D_BEV_HPP__

#include "../../Include/Interface/ExportSparse4D.h"
#include "Sparse4D_conf.pb.h"
#include <memory>
#include <string>

namespace sparse4d {
namespace bev {

// Forward declaration of internal implementation interface
class ISparse4DImpl;

class CoreImplement : public ICore {
public:
    CoreImplement();
    virtual ~CoreImplement();
    
    // ICore interface implementation
    bool initAlgorithm(const std::string exe_path, const AlgCallback& alg_cb, void* hd) override;
    bool update(void* p_pParam) override;
    void runAlgorithm(void* p_pSrcData) override;

private:
    std::unique_ptr<ISparse4DImpl> impl_;
    AlgCallback alg_cb_;
    void* user_handle_ = nullptr;
};

} // namespace bev
} // namespace sparse4d

#endif // __SPARSE4D_BEV_HPP__

