/*****************************************************************************
 IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING. By
downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install, copy or
use the software.

Contributors License Agreement

Copyright© 2007, Akhmed Umyarov. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.
- The name of Contributor may not be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
All information provided related to future Intel products and plans is
preliminary and subject to change at any time, without notice.
*****************************************************************************/
/*!\file
 * \brief Declaration of source plane object
 * \author Akhmed Umyarov
 * \date 2007
 */

#ifndef CVSOURCEPLANE_H
#define CVSOURCEPLANE_H

#include <opencv/cv.h>
#include <string>
#include "cvgenericplane.h"

//! The class represents the source image as a plane in the network
/*!
 * The class is essentially a data abstraction
 * representing image source as a dummy plane.
 * 
 * Each plane object represents one neuron (not a layer!).
 * The feature map of the source plane represents just a source image
 */
class CvSourcePlane : public CvGenericPlane
{
public:

		// Constructors/Destructors
		//  
		//! Constructor
		CvSourcePlane (std::string id, CvSize fmapsz) ;

		//! Destructor
		virtual ~CvSourcePlane ( );

		//! Dummy Forward propagation 
		virtual CvMat * fprop ( );


		//! Produces string representation of the source plane
		virtual std::string toString ( );

};

#endif // CVSOURCEPLANE_H
