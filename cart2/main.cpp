#include <wx/wx.h>
#include "CartPoleFrame.hpp"

class MyApp : public wxApp {
public:
    virtual bool OnInit() {
        auto* frame = new CartPoleFrame("CartPole RL");
        frame->Show(true);
        return true;
    }
};

wxIMPLEMENT_APP(MyApp);
