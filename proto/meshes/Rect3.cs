using pasim.math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace meshes
{

    public class Rect3
    {

        /*
            checks if point lies withing a given quadratic rectangle.
            only considers x and y
        */
        public static bool contains2(Vector3 center, float halfwidth, Vector3 point)
        {
            if (point.x < center.x - halfwidth)
                return false;

            if (point.x > center.x + halfwidth)
                return false;

            if (point.y < center.y - halfwidth)
                return false;

            if (point.y > center.y + halfwidth)
                return false;

            return true;
        }
    }
}
