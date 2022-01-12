# ML43DG-Two-shot-Spatially-varying-BRDF-and-Shape-Estimation
## A culple of best practices when transcribing from tf to torch:
- tf.variable_scope("enc") -> please add all these section marked with a comment in the torch code i.e. "# enc:" in this case (if the code is added at another position, add the tag there somewhere)
- For methods/functions that are ported using the same name: please add a docstring containing "@ DONE" so that one knows this function was already checked. 


## Reference Papers:
Two-shot-Spatially-varying-BRDF-and-Shape-Estimation: https://markboss.me/publication/cvpr20-two-shot-brdf/

Code: https://github.com/NVlabs/two-shot-brdf-shape

https://ieeexplore.ieee.org/document/9577945/media#media

https://ieeexplore.ieee.org/document/8954318

https://ieeexplore.ieee.org/document/8281537

https://ieeexplore.ieee.org/document/9157638

https://www.frontiersin.org/articles/10.3389/frobt.2020.00052/full

https://arxiv.org/abs/2110.08861

https://arxiv.org/abs/2010.03592

https://patents.google.com/patent/US7738725B2/en

https://arxiv.org/pdf/1704.01085.pdf
