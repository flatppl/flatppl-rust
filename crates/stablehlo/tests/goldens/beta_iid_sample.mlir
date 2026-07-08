module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1 = stablehlo.constant dense<3.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.compare LT, %0, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.add %0, %4 : tensor<f32>
    %7 = stablehlo.select %5, %6, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %8 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %9 = stablehlo.subtract %7, %8 : tensor<f32>
    %10 = stablehlo.constant dense<9.0> : tensor<f32>
    %11 = stablehlo.multiply %10, %9 : tensor<f32>
    %12 = stablehlo.sqrt %11 : tensor<f32>
    %13 = stablehlo.divide %4, %12 : tensor<f32>
    %14, %15 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %16 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %17 = stablehlo.shift_right_logical %15, %16 : tensor<128x4xui32>
    %18 = stablehlo.convert %17 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %19 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<128x4xf32>
    %21 = stablehlo.constant dense<2.0> : tensor<128x4xf32>
    %22 = stablehlo.constant dense<1.0> : tensor<128x4xf32>
    %23 = stablehlo.multiply %20, %21 : tensor<128x4xf32>
    %24 = stablehlo.subtract %23, %22 : tensor<128x4xf32>
    %25 = chlo.erf_inv %24 : tensor<128x4xf32> -> tensor<128x4xf32>
    %26 = stablehlo.constant dense<1.4142135> : tensor<128x4xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128x4xf32>
    %28, %29 = stablehlo.rng_bit_generator %14, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %30 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %31 = stablehlo.shift_right_logical %29, %30 : tensor<128x4xui32>
    %32 = stablehlo.convert %31 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %33 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %34 = stablehlo.multiply %32, %33 : tensor<128x4xf32>
    %35 = stablehlo.constant dense<0> : tensor<i32>
    %36 = stablehlo.constant dense<false> : tensor<4xi1>
    %37 = stablehlo.constant dense<0.0> : tensor<4xf32>
    %41:3 = stablehlo.while(%38 = %35, %39 = %36, %40 = %37) : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %42 = stablehlo.constant dense<128> : tensor<i32>
      %43 = stablehlo.compare LT, %38, %42, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %44 = stablehlo.constant dense<true> : tensor<i1>
      %45 = stablehlo.reduce(%39 init: %44) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %46 = stablehlo.not %45 : tensor<i1>
      %47 = stablehlo.and %43, %46 : tensor<i1>
      stablehlo.return %47 : tensor<i1>
    } do {
      %48 = stablehlo.constant dense<0> : tensor<i32>
      %49 = stablehlo.dynamic_slice %27, %38, %48, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %50 = stablehlo.reshape %49 : (tensor<1x4xf32>) -> tensor<4xf32>
      %51 = stablehlo.constant dense<0> : tensor<i32>
      %52 = stablehlo.dynamic_slice %34, %38, %51, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %53 = stablehlo.reshape %52 : (tensor<1x4xf32>) -> tensor<4xf32>
      %54 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %55 = stablehlo.multiply %54, %50 : tensor<4xf32>
      %56 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %57 = stablehlo.add %56, %55 : tensor<4xf32>
      %58 = stablehlo.multiply %57, %57 : tensor<4xf32>
      %59 = stablehlo.multiply %58, %57 : tensor<4xf32>
      %60 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %61 = stablehlo.multiply %60, %59 : tensor<4xf32>
      %62 = stablehlo.constant dense<0.5> : tensor<f32>
      %63 = stablehlo.multiply %50, %50 : tensor<4xf32>
      %64 = stablehlo.broadcast_in_dim %62, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %65 = stablehlo.multiply %64, %63 : tensor<4xf32>
      %66 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %67 = stablehlo.multiply %66, %59 : tensor<4xf32>
      %68 = stablehlo.negate %67 : tensor<4xf32>
      %69 = stablehlo.log %59 : tensor<4xf32>
      %70 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %71 = stablehlo.multiply %70, %69 : tensor<4xf32>
      %72 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %73 = stablehlo.add %65, %72 : tensor<4xf32>
      %74 = stablehlo.add %73, %68 : tensor<4xf32>
      %75 = stablehlo.add %74, %71 : tensor<4xf32>
      %76 = stablehlo.log %53 : tensor<4xf32>
      %77 = stablehlo.compare LT, %76, %75 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %78 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %79 = stablehlo.compare GT, %59, %78 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %80 = stablehlo.and %77, %79 : tensor<4xi1>
      %81 = stablehlo.select %39, %40, %61 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %82 = stablehlo.or %39, %80 : tensor<4xi1>
      %83 = stablehlo.constant dense<1> : tensor<i32>
      %84 = stablehlo.add %38, %83 : tensor<i32>
      stablehlo.return %84, %82, %81 : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    }
    %85, %86 = stablehlo.rng_bit_generator %28, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %87 = stablehlo.constant dense<9> : tensor<4xui32>
    %88 = stablehlo.shift_right_logical %86, %87 : tensor<4xui32>
    %89 = stablehlo.convert %88 : (tensor<4xui32>) -> tensor<4xf32>
    %90 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %91 = stablehlo.multiply %89, %90 : tensor<4xf32>
    %92 = stablehlo.divide %4, %0 : tensor<f32>
    %93 = stablehlo.broadcast_in_dim %92, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %94 = stablehlo.power %91, %93 : tensor<4xf32>
    %95 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %96 = stablehlo.select %5, %94, %95 : (tensor<i1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %97 = stablehlo.multiply %41#2, %96 : tensor<4xf32>
    %98 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %99 = stablehlo.divide %97, %98 : tensor<4xf32>
    %100 = stablehlo.constant dense<0.0> : tensor<f32>
    %101 = stablehlo.constant dense<1.0> : tensor<f32>
    %102 = stablehlo.compare LT, %1, %101 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %103 = stablehlo.add %1, %101 : tensor<f32>
    %104 = stablehlo.select %102, %103, %1 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %105 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %106 = stablehlo.subtract %104, %105 : tensor<f32>
    %107 = stablehlo.constant dense<9.0> : tensor<f32>
    %108 = stablehlo.multiply %107, %106 : tensor<f32>
    %109 = stablehlo.sqrt %108 : tensor<f32>
    %110 = stablehlo.divide %101, %109 : tensor<f32>
    %111, %112 = stablehlo.rng_bit_generator %85, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %113 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %114 = stablehlo.shift_right_logical %112, %113 : tensor<128x4xui32>
    %115 = stablehlo.convert %114 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %116 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %117 = stablehlo.multiply %115, %116 : tensor<128x4xf32>
    %118 = stablehlo.constant dense<2.0> : tensor<128x4xf32>
    %119 = stablehlo.constant dense<1.0> : tensor<128x4xf32>
    %120 = stablehlo.multiply %117, %118 : tensor<128x4xf32>
    %121 = stablehlo.subtract %120, %119 : tensor<128x4xf32>
    %122 = chlo.erf_inv %121 : tensor<128x4xf32> -> tensor<128x4xf32>
    %123 = stablehlo.constant dense<1.4142135> : tensor<128x4xf32>
    %124 = stablehlo.multiply %122, %123 : tensor<128x4xf32>
    %125, %126 = stablehlo.rng_bit_generator %111, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %127 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %128 = stablehlo.shift_right_logical %126, %127 : tensor<128x4xui32>
    %129 = stablehlo.convert %128 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %130 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %131 = stablehlo.multiply %129, %130 : tensor<128x4xf32>
    %132 = stablehlo.constant dense<0> : tensor<i32>
    %133 = stablehlo.constant dense<false> : tensor<4xi1>
    %134 = stablehlo.constant dense<0.0> : tensor<4xf32>
    %138:3 = stablehlo.while(%135 = %132, %136 = %133, %137 = %134) : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %139 = stablehlo.constant dense<128> : tensor<i32>
      %140 = stablehlo.compare LT, %135, %139, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %141 = stablehlo.constant dense<true> : tensor<i1>
      %142 = stablehlo.reduce(%136 init: %141) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %143 = stablehlo.not %142 : tensor<i1>
      %144 = stablehlo.and %140, %143 : tensor<i1>
      stablehlo.return %144 : tensor<i1>
    } do {
      %145 = stablehlo.constant dense<0> : tensor<i32>
      %146 = stablehlo.dynamic_slice %124, %135, %145, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %147 = stablehlo.reshape %146 : (tensor<1x4xf32>) -> tensor<4xf32>
      %148 = stablehlo.constant dense<0> : tensor<i32>
      %149 = stablehlo.dynamic_slice %131, %135, %148, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %150 = stablehlo.reshape %149 : (tensor<1x4xf32>) -> tensor<4xf32>
      %151 = stablehlo.broadcast_in_dim %110, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %152 = stablehlo.multiply %151, %147 : tensor<4xf32>
      %153 = stablehlo.broadcast_in_dim %101, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %154 = stablehlo.add %153, %152 : tensor<4xf32>
      %155 = stablehlo.multiply %154, %154 : tensor<4xf32>
      %156 = stablehlo.multiply %155, %154 : tensor<4xf32>
      %157 = stablehlo.broadcast_in_dim %106, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %158 = stablehlo.multiply %157, %156 : tensor<4xf32>
      %159 = stablehlo.constant dense<0.5> : tensor<f32>
      %160 = stablehlo.multiply %147, %147 : tensor<4xf32>
      %161 = stablehlo.broadcast_in_dim %159, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %162 = stablehlo.multiply %161, %160 : tensor<4xf32>
      %163 = stablehlo.broadcast_in_dim %106, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %164 = stablehlo.multiply %163, %156 : tensor<4xf32>
      %165 = stablehlo.negate %164 : tensor<4xf32>
      %166 = stablehlo.log %156 : tensor<4xf32>
      %167 = stablehlo.broadcast_in_dim %106, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %168 = stablehlo.multiply %167, %166 : tensor<4xf32>
      %169 = stablehlo.broadcast_in_dim %106, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %170 = stablehlo.add %162, %169 : tensor<4xf32>
      %171 = stablehlo.add %170, %165 : tensor<4xf32>
      %172 = stablehlo.add %171, %168 : tensor<4xf32>
      %173 = stablehlo.log %150 : tensor<4xf32>
      %174 = stablehlo.compare LT, %173, %172 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %175 = stablehlo.broadcast_in_dim %100, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %176 = stablehlo.compare GT, %156, %175 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %177 = stablehlo.and %174, %176 : tensor<4xi1>
      %178 = stablehlo.select %136, %137, %158 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %179 = stablehlo.or %136, %177 : tensor<4xi1>
      %180 = stablehlo.constant dense<1> : tensor<i32>
      %181 = stablehlo.add %135, %180 : tensor<i32>
      stablehlo.return %181, %179, %178 : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    }
    %182, %183 = stablehlo.rng_bit_generator %125, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %184 = stablehlo.constant dense<9> : tensor<4xui32>
    %185 = stablehlo.shift_right_logical %183, %184 : tensor<4xui32>
    %186 = stablehlo.convert %185 : (tensor<4xui32>) -> tensor<4xf32>
    %187 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %188 = stablehlo.multiply %186, %187 : tensor<4xf32>
    %189 = stablehlo.divide %101, %1 : tensor<f32>
    %190 = stablehlo.broadcast_in_dim %189, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %191 = stablehlo.power %188, %190 : tensor<4xf32>
    %192 = stablehlo.broadcast_in_dim %101, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %193 = stablehlo.select %102, %191, %192 : (tensor<i1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %194 = stablehlo.multiply %138#2, %193 : tensor<4xf32>
    %195 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %196 = stablehlo.divide %194, %195 : tensor<4xf32>
    %197 = stablehlo.add %99, %196 : tensor<4xf32>
    %198 = stablehlo.divide %99, %197 : tensor<4xf32>
    return %198, %182 : tensor<4xf32>, tensor<2xui64>
  }
}
