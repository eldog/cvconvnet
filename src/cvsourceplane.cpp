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
 * \brief Implementation of source plane class
 * \author Akhmed Umyarov
 * \date 2007
 */

#include "cvsourceplane.h"
#include <iostream>
#include <sstream>

using namespace std;


// Constructors/Destructors
//  
/*!
 * Constructor creates a new source plane with specified name and of 
 * specified size.
 * \param id name of the plane
 * \param fmapsz size of the featuremap for this plane. For source planes,
 * it is just a size of image, since source plane is just an abstraction.
 */
CvSourcePlane::CvSourcePlane  (std::string id, CvSize fmapsz)
	: CvGenericPlane(id, fmapsz, fmapsz) 
{
}

CvSourcePlane::~CvSourcePlane ( ) 
{ 
}

//  
// Methods
//  

/*!
 * \return Pointer to plane's featuremap
 */
CvMat * CvSourcePlane::fprop ()
{
	return m_fmap;
}


/*! The method produces an XML representation of the complete information about 
 * the source plane.
 * The only useful information about source plane is its name and its size.
 * Weights and connections (to parents) are not used.
 * The representation does NOT include current state of the plane
 * such as its feature map.
 * The method is mainly used by the CvConvNet object.
 * \return string containing XML description of the plane
 */
string CvSourcePlane::toString ( ) 
{
	ostringstream xml;
 	xml << "\t<plane id=\"" << m_id << "\" type=\"source\" featuremapsize=\"" << m_fmapsz.width << "x" << m_fmapsz.height << "\">" << endl;
	xml << "\t</plane>" << endl;
	
	return xml.str();
}
