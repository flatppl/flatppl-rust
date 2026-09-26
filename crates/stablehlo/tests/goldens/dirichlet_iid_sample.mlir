module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<5x3xf32>, tensor<2xui64>) {
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.constant dense<0.0> : tensor<f32>
    %8 = stablehlo.constant dense<1.6666666269302368> : tensor<f32>
    %12 = stablehlo.constant dense<0.25819888710975647> : tensor<f32>
    %13, %14 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %15 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %16 = stablehlo.shift_right_logical %14, %15 : tensor<128x5xui32>
    %17 = stablehlo.convert %16 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %18 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128x5xf32>
    %20 = stablehlo.constant dense<2.0> : tensor<128x5xf32>
    %21 = stablehlo.constant dense<1.0> : tensor<128x5xf32>
    %22 = stablehlo.multiply %19, %20 : tensor<128x5xf32>
    %23 = stablehlo.subtract %22, %21 : tensor<128x5xf32>
    %24 = chlo.erf_inv %23 : tensor<128x5xf32> -> tensor<128x5xf32>
    %25 = stablehlo.constant dense<1.4142135> : tensor<128x5xf32>
    %26 = stablehlo.multiply %24, %25 : tensor<128x5xf32>
    %27, %28 = stablehlo.rng_bit_generator %13, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %29 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %30 = stablehlo.shift_right_logical %28, %29 : tensor<128x5xui32>
    %31 = stablehlo.convert %30 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %32 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %33 = stablehlo.multiply %31, %32 : tensor<128x5xf32>
    %34 = stablehlo.constant dense<0> : tensor<i32>
    %35 = stablehlo.constant dense<false> : tensor<5xi1>
    %36 = stablehlo.constant dense<0.0> : tensor<5xf32>
    %40:3 = stablehlo.while(%37 = %34, %38 = %35, %39 = %36) : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    cond {
      %41 = stablehlo.constant dense<128> : tensor<i32>
      %42 = stablehlo.compare LT, %37, %41, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %43 = stablehlo.constant dense<true> : tensor<i1>
      %44 = stablehlo.reduce(%38 init: %43) applies stablehlo.and across dimensions = [0] : (tensor<5xi1>, tensor<i1>) -> tensor<i1>
      %45 = stablehlo.not %44 : tensor<i1>
      %46 = stablehlo.and %42, %45 : tensor<i1>
      stablehlo.return %46 : tensor<i1>
    } do {
      %47 = stablehlo.constant dense<0> : tensor<i32>
      %48 = stablehlo.dynamic_slice %26, %37, %47, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %49 = stablehlo.reshape %48 : (tensor<1x5xf32>) -> tensor<5xf32>
      %50 = stablehlo.dynamic_slice %33, %37, %47, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %51 = stablehlo.reshape %50 : (tensor<1x5xf32>) -> tensor<5xf32>
      %52 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %53 = stablehlo.multiply %52, %49 : tensor<5xf32>
      %54 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %55 = stablehlo.add %54, %53 : tensor<5xf32>
      %56 = stablehlo.multiply %55, %55 : tensor<5xf32>
      %57 = stablehlo.multiply %56, %55 : tensor<5xf32>
      %58 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %59 = stablehlo.multiply %58, %57 : tensor<5xf32>
      %60 = stablehlo.constant dense<0.5> : tensor<f32>
      %61 = stablehlo.multiply %49, %49 : tensor<5xf32>
      %62 = stablehlo.broadcast_in_dim %60, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %63 = stablehlo.multiply %62, %61 : tensor<5xf32>
      %64 = stablehlo.negate %59 : tensor<5xf32>
      %65 = stablehlo.log %57 : tensor<5xf32>
      %66 = stablehlo.multiply %58, %65 : tensor<5xf32>
      %67 = stablehlo.add %63, %58 : tensor<5xf32>
      %68 = stablehlo.add %67, %64 : tensor<5xf32>
      %69 = stablehlo.add %68, %66 : tensor<5xf32>
      %70 = stablehlo.log %51 : tensor<5xf32>
      %71 = stablehlo.compare LT, %70, %69 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %72 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %73 = stablehlo.compare GT, %57, %72 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %74 = stablehlo.and %71, %73 : tensor<5xi1>
      %75 = stablehlo.select %38, %39, %59 : (tensor<5xi1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
      %76 = stablehlo.or %38, %74 : tensor<5xi1>
      %77 = stablehlo.constant dense<1> : tensor<i32>
      %78 = stablehlo.add %37, %77 : tensor<i32>
      stablehlo.return %78, %76, %75 : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    }
    %79, %80 = stablehlo.rng_bit_generator %27, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<5xui32>)
    %81 = stablehlo.constant dense<9> : tensor<5xui32>
    %82 = stablehlo.shift_right_logical %80, %81 : tensor<5xui32>
    %83 = stablehlo.convert %82 : (tensor<5xui32>) -> tensor<5xf32>
    %84 = stablehlo.constant dense<1.1920929E-7> : tensor<5xf32>
    %85 = stablehlo.multiply %83, %84 : tensor<5xf32>
    %89 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %90 = stablehlo.multiply %40#2, %89 : tensor<5xf32>
    %91 = stablehlo.divide %90, %89 : tensor<5xf32>
    %95 = stablehlo.constant dense<2.6666667461395264> : tensor<f32>
    %98 = stablehlo.constant dense<0.20412413775920868> : tensor<f32>
    %99, %100 = stablehlo.rng_bit_generator %79, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %101 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %102 = stablehlo.shift_right_logical %100, %101 : tensor<128x5xui32>
    %103 = stablehlo.convert %102 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %104 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %105 = stablehlo.multiply %103, %104 : tensor<128x5xf32>
    %106 = stablehlo.multiply %105, %20 : tensor<128x5xf32>
    %107 = stablehlo.subtract %106, %21 : tensor<128x5xf32>
    %108 = chlo.erf_inv %107 : tensor<128x5xf32> -> tensor<128x5xf32>
    %109 = stablehlo.constant dense<1.4142135> : tensor<128x5xf32>
    %110 = stablehlo.multiply %108, %109 : tensor<128x5xf32>
    %111, %112 = stablehlo.rng_bit_generator %99, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %113 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %114 = stablehlo.shift_right_logical %112, %113 : tensor<128x5xui32>
    %115 = stablehlo.convert %114 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %116 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %117 = stablehlo.multiply %115, %116 : tensor<128x5xf32>
    %118 = stablehlo.constant dense<false> : tensor<5xi1>
    %122:3 = stablehlo.while(%119 = %34, %120 = %118, %121 = %36) : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    cond {
      %123 = stablehlo.constant dense<128> : tensor<i32>
      %124 = stablehlo.compare LT, %119, %123, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %125 = stablehlo.constant dense<true> : tensor<i1>
      %126 = stablehlo.reduce(%120 init: %125) applies stablehlo.and across dimensions = [0] : (tensor<5xi1>, tensor<i1>) -> tensor<i1>
      %127 = stablehlo.not %126 : tensor<i1>
      %128 = stablehlo.and %124, %127 : tensor<i1>
      stablehlo.return %128 : tensor<i1>
    } do {
      %129 = stablehlo.constant dense<0> : tensor<i32>
      %130 = stablehlo.dynamic_slice %110, %119, %129, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %131 = stablehlo.reshape %130 : (tensor<1x5xf32>) -> tensor<5xf32>
      %132 = stablehlo.dynamic_slice %117, %119, %129, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %133 = stablehlo.reshape %132 : (tensor<1x5xf32>) -> tensor<5xf32>
      %134 = stablehlo.broadcast_in_dim %98, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %135 = stablehlo.multiply %134, %131 : tensor<5xf32>
      %136 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %137 = stablehlo.add %136, %135 : tensor<5xf32>
      %138 = stablehlo.multiply %137, %137 : tensor<5xf32>
      %139 = stablehlo.multiply %138, %137 : tensor<5xf32>
      %140 = stablehlo.broadcast_in_dim %95, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %141 = stablehlo.multiply %140, %139 : tensor<5xf32>
      %142 = stablehlo.constant dense<0.5> : tensor<f32>
      %143 = stablehlo.multiply %131, %131 : tensor<5xf32>
      %144 = stablehlo.broadcast_in_dim %142, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %145 = stablehlo.multiply %144, %143 : tensor<5xf32>
      %146 = stablehlo.negate %141 : tensor<5xf32>
      %147 = stablehlo.log %139 : tensor<5xf32>
      %148 = stablehlo.multiply %140, %147 : tensor<5xf32>
      %149 = stablehlo.add %145, %140 : tensor<5xf32>
      %150 = stablehlo.add %149, %146 : tensor<5xf32>
      %151 = stablehlo.add %150, %148 : tensor<5xf32>
      %152 = stablehlo.log %133 : tensor<5xf32>
      %153 = stablehlo.compare LT, %152, %151 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %154 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %155 = stablehlo.compare GT, %139, %154 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %156 = stablehlo.and %153, %155 : tensor<5xi1>
      %157 = stablehlo.select %120, %121, %141 : (tensor<5xi1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
      %158 = stablehlo.or %120, %156 : tensor<5xi1>
      %159 = stablehlo.constant dense<1> : tensor<i32>
      %160 = stablehlo.add %119, %159 : tensor<i32>
      stablehlo.return %160, %158, %157 : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    }
    %161, %162 = stablehlo.rng_bit_generator %111, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<5xui32>)
    %163 = stablehlo.constant dense<9> : tensor<5xui32>
    %164 = stablehlo.shift_right_logical %162, %163 : tensor<5xui32>
    %165 = stablehlo.convert %164 : (tensor<5xui32>) -> tensor<5xf32>
    %166 = stablehlo.constant dense<1.1920929E-7> : tensor<5xf32>
    %167 = stablehlo.multiply %165, %166 : tensor<5xf32>
    %171 = stablehlo.multiply %122#2, %89 : tensor<5xf32>
    %172 = stablehlo.divide %171, %89 : tensor<5xf32>
    %176 = stablehlo.constant dense<3.6666667461395264> : tensor<f32>
    %179 = stablehlo.constant dense<0.17407765984535217> : tensor<f32>
    %180, %181 = stablehlo.rng_bit_generator %161, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %182 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %183 = stablehlo.shift_right_logical %181, %182 : tensor<128x5xui32>
    %184 = stablehlo.convert %183 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %185 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %186 = stablehlo.multiply %184, %185 : tensor<128x5xf32>
    %187 = stablehlo.multiply %186, %20 : tensor<128x5xf32>
    %188 = stablehlo.subtract %187, %21 : tensor<128x5xf32>
    %189 = chlo.erf_inv %188 : tensor<128x5xf32> -> tensor<128x5xf32>
    %190 = stablehlo.constant dense<1.4142135> : tensor<128x5xf32>
    %191 = stablehlo.multiply %189, %190 : tensor<128x5xf32>
    %192, %193 = stablehlo.rng_bit_generator %180, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %194 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %195 = stablehlo.shift_right_logical %193, %194 : tensor<128x5xui32>
    %196 = stablehlo.convert %195 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %197 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %198 = stablehlo.multiply %196, %197 : tensor<128x5xf32>
    %199 = stablehlo.constant dense<false> : tensor<5xi1>
    %203:3 = stablehlo.while(%200 = %34, %201 = %199, %202 = %36) : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    cond {
      %204 = stablehlo.constant dense<128> : tensor<i32>
      %205 = stablehlo.compare LT, %200, %204, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %206 = stablehlo.constant dense<true> : tensor<i1>
      %207 = stablehlo.reduce(%201 init: %206) applies stablehlo.and across dimensions = [0] : (tensor<5xi1>, tensor<i1>) -> tensor<i1>
      %208 = stablehlo.not %207 : tensor<i1>
      %209 = stablehlo.and %205, %208 : tensor<i1>
      stablehlo.return %209 : tensor<i1>
    } do {
      %210 = stablehlo.constant dense<0> : tensor<i32>
      %211 = stablehlo.dynamic_slice %191, %200, %210, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %212 = stablehlo.reshape %211 : (tensor<1x5xf32>) -> tensor<5xf32>
      %213 = stablehlo.dynamic_slice %198, %200, %210, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %214 = stablehlo.reshape %213 : (tensor<1x5xf32>) -> tensor<5xf32>
      %215 = stablehlo.broadcast_in_dim %179, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %216 = stablehlo.multiply %215, %212 : tensor<5xf32>
      %217 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %218 = stablehlo.add %217, %216 : tensor<5xf32>
      %219 = stablehlo.multiply %218, %218 : tensor<5xf32>
      %220 = stablehlo.multiply %219, %218 : tensor<5xf32>
      %221 = stablehlo.broadcast_in_dim %176, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %222 = stablehlo.multiply %221, %220 : tensor<5xf32>
      %223 = stablehlo.constant dense<0.5> : tensor<f32>
      %224 = stablehlo.multiply %212, %212 : tensor<5xf32>
      %225 = stablehlo.broadcast_in_dim %223, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %226 = stablehlo.multiply %225, %224 : tensor<5xf32>
      %227 = stablehlo.negate %222 : tensor<5xf32>
      %228 = stablehlo.log %220 : tensor<5xf32>
      %229 = stablehlo.multiply %221, %228 : tensor<5xf32>
      %230 = stablehlo.add %226, %221 : tensor<5xf32>
      %231 = stablehlo.add %230, %227 : tensor<5xf32>
      %232 = stablehlo.add %231, %229 : tensor<5xf32>
      %233 = stablehlo.log %214 : tensor<5xf32>
      %234 = stablehlo.compare LT, %233, %232 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %235 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %236 = stablehlo.compare GT, %220, %235 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %237 = stablehlo.and %234, %236 : tensor<5xi1>
      %238 = stablehlo.select %201, %202, %222 : (tensor<5xi1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
      %239 = stablehlo.or %201, %237 : tensor<5xi1>
      %240 = stablehlo.constant dense<1> : tensor<i32>
      %241 = stablehlo.add %200, %240 : tensor<i32>
      stablehlo.return %241, %239, %238 : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    }
    %242, %243 = stablehlo.rng_bit_generator %192, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<5xui32>)
    %244 = stablehlo.constant dense<9> : tensor<5xui32>
    %245 = stablehlo.shift_right_logical %243, %244 : tensor<5xui32>
    %246 = stablehlo.convert %245 : (tensor<5xui32>) -> tensor<5xf32>
    %247 = stablehlo.constant dense<1.1920929E-7> : tensor<5xf32>
    %248 = stablehlo.multiply %246, %247 : tensor<5xf32>
    %252 = stablehlo.multiply %203#2, %89 : tensor<5xf32>
    %253 = stablehlo.divide %252, %89 : tensor<5xf32>
    %254 = stablehlo.reshape %91 : (tensor<5xf32>) -> tensor<1x5xf32>
    %255 = stablehlo.reshape %172 : (tensor<5xf32>) -> tensor<1x5xf32>
    %256 = stablehlo.reshape %253 : (tensor<5xf32>) -> tensor<1x5xf32>
    %257 = stablehlo.concatenate %254, %255, %256, dim = 0 : (tensor<1x5xf32>, tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<3x5xf32>
    %258 = stablehlo.transpose %257, dims = [1, 0] : (tensor<3x5xf32>) -> tensor<5x3xf32>
    %259 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %260 = stablehlo.reduce(%258 init: %259) applies stablehlo.add across dimensions = [1] : (tensor<5x3xf32>, tensor<f32>) -> tensor<5xf32>
    %261 = stablehlo.broadcast_in_dim %260, dims = [0] : (tensor<5xf32>) -> tensor<5x3xf32>
    %262 = stablehlo.divide %258, %261 : tensor<5x3xf32>
    return %262, %242 : tensor<5x3xf32>, tensor<2xui64>
  }
}
