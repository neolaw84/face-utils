# face-extractor

Small code-base to extract face box based on dlib.

Updated the code-base to extract in more manners based on insightface.

# Changelog

## 2021-08-22 : Face Detection by Insightface

* rename the repo to `face_utils`
* added `face_utils` package
* extractor now uses [insightface](https://pypi.org/project/insightface/) for detection
* formal formulation of `extract` function
* addition of rotation correction (w.r.t the face's Z axis, i.e. from the nose to back of head). 

# Copy-rights etc.

The [test image](https://live.staticflickr.com/2605/3721476240_bf643c709e.jpg) retrived from https://live.staticflickr.com is attributed as follow:

> "Models photo shoot" by davidyuweb is licensed under CC BY-NC 2.0

The [test image](https://www.pexels.com/photo/back-view-of-a-woman-in-brown-dress-3866555/) is attributed as follow:

> All photos and videos on Pexels are free to use.
> Attribution is not required. Giving credit to the photographer or Pexels is not necessary but always appreciated.
> You can modify the photos and videos from Pexels. Be creative and edit them as you like. 
> by [Pexels.com's license](https://www.pexels.com/license/)

The [test image](https://unsplash.com/photos/6xv4A1VA1rU) is attributed as follow:

> All photos can be downloaded and used for free
> Commercial and non-commercial purposes
> No permission needed (though attribution is appreciated!)
> What is not permitted 👎
> Photos cannot be sold without significant modification.
> Compiling photos from Unsplash to replicate a similar or competing service.
> by [Unsplash.com's license](https://unsplash.com/license)
 
The [dlib's 68 face landmarks](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) retrived from http://dlib.net is attributed as follow:

> Boost Software License - Version 1.0 - August 17th, 2003

For more information, refer to [dlib's License here](http://dlib.net/license.html).

The [insightface's code](https://pypi.org/project/insightface/) is attributed as follow:

> MIT License 

The [insightface's models](https://pypi.org/project/insightface/), which the above code automatically downloads is governed by: 

> Non commercial License

For more information, refer to [insightface github page](https://github.com/deepinsight/insightface). 