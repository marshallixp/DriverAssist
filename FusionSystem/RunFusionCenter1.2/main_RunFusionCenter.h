//
//  main_RunFusionCenter.h
//  RunFusionCenter1.2
//
//  Created by XUANPENG LI on 11/11/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef RunFusionCenter_main_RunFusionCenter_h
#define RunFusionCenter_main_RunFusionCenter_h

namespace RunFusionCenter {
    enum CombineType { 
        Single_ORD,
        Single_ORD_TEF,
        Multi_ORD,
        Multi_Fusion
    };
    
    int Process (int argc, const char * argv[]);
}

#endif
