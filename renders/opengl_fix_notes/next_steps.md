
Im seeing an error in the visualization with the opengl viewer, i run the example in example/ellipse_interctive.py and i see an strange “scalar field” artifacts inside the ellipse indepent of the `ray_width`, images of such strange artifacts are in renders.  



Please review carefully the 2D OpenGL viewer in `raytracer/visualization_opengl.py` and find what is wrong here such that the error that we are seeing is producted, if you find why propose the changes that solves the problem. 


- Keep `ray_width` and `sigma_factor` unchanged.
- The solution must work during interactive pan/zoom and while redrawing cached rays.

Try of not simple "add code" but really find the problem a propose a real solution, tiny refactors in post of order, clarify, modularity in the code are allowed if are needed. 

in /renders/opengl_fix_notes there are images of the problems, the main one is the strange curves/scalarfileds that are appering n the image in all the ellise,this strange curves looks curved near to the perimeter of the sphere, the other important thing is that the source is not showing complete, the rays after some angle behaves strange and the ray density decrease as you can see in the near_source images. 



