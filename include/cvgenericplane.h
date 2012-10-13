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
 * \brief Declaration of generic plane interface
 * \author Akhmed Umyarov
 * \date 2007
 */

#ifndef CVGENERICPLANE_H
#define CVGENERICPLANE_H

#include <opencv/cv.h>
#include <string>
#include <vector>

//! The class provides a generic interface that every plane (neuron) must implement
/*! The class provides a generic interface that every plane must implement, 
 * it also provides basic functionality that is used by every type of the plane
 * 
 * Each plane object represents one neuron (not a layer!).
 * The object also contains weights for this neuron and feature map
 * for this neuron.
 */
class CvGenericPlane
{
public:

		// Constructors/Destructors
		//  
		//! Constructor
		CvGenericPlane (std::string id, CvSize fmapsz, CvSize neurosz);
		
		//! Destructor
		virtual ~CvGenericPlane ( );

		//! Connect the plane to the parent planes
		int connto(std::vector<CvGenericPlane *> &pplane);

		//! Connect the plane back to child
		int connchild(CvGenericPlane *cplane);

		//! Disconnect the plane completely
		int disconn();

		//! Do forward propagation
		virtual CvMat * fprop ( ) = 0;

		// Do backward error propagation
// 		virtual CvMat * bprop ( ) = 0;

		//! Produce string representation
		virtual std::string toString ( ) = 0;

		//! Explicitly set the weights for the plane's neuron
		virtual int setweight(std::vector<double> &weights);

		//! Get a pointer to plane's feature map
		CvMat * getfmap ( );

		//! Explicitly set values for plane's feature map
		int setfmap ( CvArr * source );

		//! Get plane's text id
		std::string getid();

protected:
		std::string m_id; //!< Plane string id
		std::vector<CvGenericPlane *> m_pplane; //!< Links to parents (for fprop)
		std::vector<CvGenericPlane *> m_cplane; //!< Links to childs (for bprop)
		std::vector<CvMat *> m_pfmap; //!< Cached pointers to parents feature maps (for fprop)
		std::vector<double> m_delta; //!< Deltas (will be used for bprop)
	
		CvMat *m_fmap; //!< Container of feature map for this plane
		CvSize m_fmapsz; //!< Size of feature map
		CvSize m_neurosz;//!< Neuron window

		std::vector<double> m_weight; //!< Container for weights of plane's neuron 
		int m_connected; //!< Flag specifying whether we are already connected to parents or not
};

#endif // CVGENERICPLANE_H
