using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.net
{

    public class Assert
    {

        public static void NotNull(object o) 
        {
            if (o == null)
                throw new Exception("object is null");
        }

        public static void Null(object o)
        {
            if (o != null)
                throw new Exception("object is not null");
        }
    }
}
