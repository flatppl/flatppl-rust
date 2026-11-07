module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.constant dense<false> : tensor<i1>
    %7 = stablehlo.constant dense<"0x5555D53F"> : tensor<f32>
    %11 = stablehlo.constant dense<"0xA532843E"> : tensor<f32>
    %12, %13 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %14 = stablehlo.constant dense<9> : tensor<128xui32>
    %15 = stablehlo.shift_right_logical %13, %14 : tensor<128xui32>
    %16 = stablehlo.convert %15 : (tensor<128xui32>) -> tensor<128xf32>
    %17 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<128xf32>
    %19 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %20 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %21 = stablehlo.multiply %18, %19 : tensor<128xf32>
    %22 = stablehlo.subtract %21, %20 : tensor<128xf32>
    %23 = chlo.erf_inv %22 : tensor<128xf32> -> tensor<128xf32>
    %24 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %25 = stablehlo.multiply %23, %24 : tensor<128xf32>
    %26, %27 = stablehlo.rng_bit_generator %12, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %28 = stablehlo.constant dense<9> : tensor<128xui32>
    %29 = stablehlo.shift_right_logical %27, %28 : tensor<128xui32>
    %30 = stablehlo.convert %29 : (tensor<128xui32>) -> tensor<128xf32>
    %31 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %32 = stablehlo.multiply %30, %31 : tensor<128xf32>
    %33 = stablehlo.constant dense<0> : tensor<i32>
    %37:3 = stablehlo.while(%34 = %33, %35 = %4, %36 = %3) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %38 = stablehlo.constant dense<128> : tensor<i32>
      %39 = stablehlo.compare LT, %34, %38, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %40 = stablehlo.not %35 : tensor<i1>
      %41 = stablehlo.and %40, %39 : tensor<i1>
      stablehlo.return %41 : tensor<i1>
    } do {
      %42 = stablehlo.dynamic_slice %25, %34, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %43 = stablehlo.reshape %42 : (tensor<1xf32>) -> tensor<f32>
      %44 = stablehlo.dynamic_slice %32, %34, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %45 = stablehlo.reshape %44 : (tensor<1xf32>) -> tensor<f32>
      %46 = stablehlo.multiply %11, %43 : tensor<f32>
      %47 = stablehlo.add %2, %46 : tensor<f32>
      %48 = stablehlo.multiply %47, %47 : tensor<f32>
      %49 = stablehlo.multiply %48, %47 : tensor<f32>
      %50 = stablehlo.multiply %7, %49 : tensor<f32>
      %51 = stablehlo.constant dense<0.5> : tensor<f32>
      %52 = stablehlo.multiply %43, %43 : tensor<f32>
      %53 = stablehlo.multiply %51, %52 : tensor<f32>
      %54 = stablehlo.negate %50 : tensor<f32>
      %55 = stablehlo.log %49 : tensor<f32>
      %56 = stablehlo.multiply %7, %55 : tensor<f32>
      %57 = stablehlo.add %53, %7 : tensor<f32>
      %58 = stablehlo.add %57, %54 : tensor<f32>
      %59 = stablehlo.add %58, %56 : tensor<f32>
      %60 = stablehlo.log %45 : tensor<f32>
      %61 = stablehlo.compare LT, %60, %59 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %62 = stablehlo.compare GT, %49, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %63 = stablehlo.and %61, %62 : tensor<i1>
      %64 = stablehlo.constant dense<1> : tensor<i32>
      %65 = stablehlo.add %34, %64 : tensor<i32>
      stablehlo.return %65, %63, %50 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %66, %67 = stablehlo.rng_bit_generator %26, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %68 = stablehlo.constant dense<9> : tensor<ui32>
    %69 = stablehlo.shift_right_logical %67, %68 : tensor<ui32>
    %70 = stablehlo.convert %69 : (tensor<ui32>) -> tensor<f32>
    %71 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %72 = stablehlo.multiply %70, %71 : tensor<f32>
    %73 = stablehlo.log %37#2 : tensor<f32>
    %76 = stablehlo.add %73, %3 : tensor<f32>
    %77 = stablehlo.constant dense<"0x00000000"> : tensor<f32>
    %78 = stablehlo.subtract %76, %77 : tensor<f32>
    %80 = stablehlo.constant dense<"0xABAA2A40"> : tensor<f32>
    %83 = stablehlo.constant dense<"0xEB05513E"> : tensor<f32>
    %84, %85 = stablehlo.rng_bit_generator %66, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %86 = stablehlo.constant dense<9> : tensor<128xui32>
    %87 = stablehlo.shift_right_logical %85, %86 : tensor<128xui32>
    %88 = stablehlo.convert %87 : (tensor<128xui32>) -> tensor<128xf32>
    %89 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %90 = stablehlo.multiply %88, %89 : tensor<128xf32>
    %91 = stablehlo.multiply %90, %19 : tensor<128xf32>
    %92 = stablehlo.subtract %91, %20 : tensor<128xf32>
    %93 = chlo.erf_inv %92 : tensor<128xf32> -> tensor<128xf32>
    %94 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %95 = stablehlo.multiply %93, %94 : tensor<128xf32>
    %96, %97 = stablehlo.rng_bit_generator %84, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %98 = stablehlo.constant dense<9> : tensor<128xui32>
    %99 = stablehlo.shift_right_logical %97, %98 : tensor<128xui32>
    %100 = stablehlo.convert %99 : (tensor<128xui32>) -> tensor<128xf32>
    %101 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %102 = stablehlo.multiply %100, %101 : tensor<128xf32>
    %106:3 = stablehlo.while(%103 = %33, %104 = %4, %105 = %3) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %107 = stablehlo.constant dense<128> : tensor<i32>
      %108 = stablehlo.compare LT, %103, %107, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %109 = stablehlo.not %104 : tensor<i1>
      %110 = stablehlo.and %109, %108 : tensor<i1>
      stablehlo.return %110 : tensor<i1>
    } do {
      %111 = stablehlo.dynamic_slice %95, %103, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %112 = stablehlo.reshape %111 : (tensor<1xf32>) -> tensor<f32>
      %113 = stablehlo.dynamic_slice %102, %103, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %114 = stablehlo.reshape %113 : (tensor<1xf32>) -> tensor<f32>
      %115 = stablehlo.multiply %83, %112 : tensor<f32>
      %116 = stablehlo.add %2, %115 : tensor<f32>
      %117 = stablehlo.multiply %116, %116 : tensor<f32>
      %118 = stablehlo.multiply %117, %116 : tensor<f32>
      %119 = stablehlo.multiply %80, %118 : tensor<f32>
      %120 = stablehlo.constant dense<0.5> : tensor<f32>
      %121 = stablehlo.multiply %112, %112 : tensor<f32>
      %122 = stablehlo.multiply %120, %121 : tensor<f32>
      %123 = stablehlo.negate %119 : tensor<f32>
      %124 = stablehlo.log %118 : tensor<f32>
      %125 = stablehlo.multiply %80, %124 : tensor<f32>
      %126 = stablehlo.add %122, %80 : tensor<f32>
      %127 = stablehlo.add %126, %123 : tensor<f32>
      %128 = stablehlo.add %127, %125 : tensor<f32>
      %129 = stablehlo.log %114 : tensor<f32>
      %130 = stablehlo.compare LT, %129, %128 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %131 = stablehlo.compare GT, %118, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %132 = stablehlo.and %130, %131 : tensor<i1>
      %133 = stablehlo.constant dense<1> : tensor<i32>
      %134 = stablehlo.add %103, %133 : tensor<i32>
      stablehlo.return %134, %132, %119 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %135, %136 = stablehlo.rng_bit_generator %96, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %137 = stablehlo.constant dense<9> : tensor<ui32>
    %138 = stablehlo.shift_right_logical %136, %137 : tensor<ui32>
    %139 = stablehlo.convert %138 : (tensor<ui32>) -> tensor<f32>
    %140 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %141 = stablehlo.multiply %139, %140 : tensor<f32>
    %142 = stablehlo.log %106#2 : tensor<f32>
    %145 = stablehlo.add %142, %3 : tensor<f32>
    %146 = stablehlo.subtract %145, %77 : tensor<f32>
    %147 = stablehlo.subtract %146, %78 : tensor<f32>
    %148 = stablehlo.exponential %147 : tensor<f32>
    %149 = stablehlo.add %2, %148 : tensor<f32>
    %150 = stablehlo.divide %2, %149 : tensor<f32>
    return %150, %135 : tensor<f32>, tensor<2xui64>
  }
}
