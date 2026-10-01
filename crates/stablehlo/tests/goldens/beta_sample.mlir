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
    %75 = stablehlo.multiply %37#2, %2 : tensor<f32>
    %76 = stablehlo.divide %75, %2 : tensor<f32>
    %78 = stablehlo.constant dense<"0xABAA2A40"> : tensor<f32>
    %81 = stablehlo.constant dense<"0xEB05513E"> : tensor<f32>
    %82, %83 = stablehlo.rng_bit_generator %66, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %84 = stablehlo.constant dense<9> : tensor<128xui32>
    %85 = stablehlo.shift_right_logical %83, %84 : tensor<128xui32>
    %86 = stablehlo.convert %85 : (tensor<128xui32>) -> tensor<128xf32>
    %87 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %88 = stablehlo.multiply %86, %87 : tensor<128xf32>
    %89 = stablehlo.multiply %88, %19 : tensor<128xf32>
    %90 = stablehlo.subtract %89, %20 : tensor<128xf32>
    %91 = chlo.erf_inv %90 : tensor<128xf32> -> tensor<128xf32>
    %92 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %93 = stablehlo.multiply %91, %92 : tensor<128xf32>
    %94, %95 = stablehlo.rng_bit_generator %82, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %96 = stablehlo.constant dense<9> : tensor<128xui32>
    %97 = stablehlo.shift_right_logical %95, %96 : tensor<128xui32>
    %98 = stablehlo.convert %97 : (tensor<128xui32>) -> tensor<128xf32>
    %99 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %100 = stablehlo.multiply %98, %99 : tensor<128xf32>
    %104:3 = stablehlo.while(%101 = %33, %102 = %4, %103 = %3) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %105 = stablehlo.constant dense<128> : tensor<i32>
      %106 = stablehlo.compare LT, %101, %105, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %107 = stablehlo.not %102 : tensor<i1>
      %108 = stablehlo.and %107, %106 : tensor<i1>
      stablehlo.return %108 : tensor<i1>
    } do {
      %109 = stablehlo.dynamic_slice %93, %101, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %110 = stablehlo.reshape %109 : (tensor<1xf32>) -> tensor<f32>
      %111 = stablehlo.dynamic_slice %100, %101, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %112 = stablehlo.reshape %111 : (tensor<1xf32>) -> tensor<f32>
      %113 = stablehlo.multiply %81, %110 : tensor<f32>
      %114 = stablehlo.add %2, %113 : tensor<f32>
      %115 = stablehlo.multiply %114, %114 : tensor<f32>
      %116 = stablehlo.multiply %115, %114 : tensor<f32>
      %117 = stablehlo.multiply %78, %116 : tensor<f32>
      %118 = stablehlo.constant dense<0.5> : tensor<f32>
      %119 = stablehlo.multiply %110, %110 : tensor<f32>
      %120 = stablehlo.multiply %118, %119 : tensor<f32>
      %121 = stablehlo.negate %117 : tensor<f32>
      %122 = stablehlo.log %116 : tensor<f32>
      %123 = stablehlo.multiply %78, %122 : tensor<f32>
      %124 = stablehlo.add %120, %78 : tensor<f32>
      %125 = stablehlo.add %124, %121 : tensor<f32>
      %126 = stablehlo.add %125, %123 : tensor<f32>
      %127 = stablehlo.log %112 : tensor<f32>
      %128 = stablehlo.compare LT, %127, %126 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %129 = stablehlo.compare GT, %116, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %130 = stablehlo.and %128, %129 : tensor<i1>
      %131 = stablehlo.constant dense<1> : tensor<i32>
      %132 = stablehlo.add %101, %131 : tensor<i32>
      stablehlo.return %132, %130, %117 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %133, %134 = stablehlo.rng_bit_generator %94, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %135 = stablehlo.constant dense<9> : tensor<ui32>
    %136 = stablehlo.shift_right_logical %134, %135 : tensor<ui32>
    %137 = stablehlo.convert %136 : (tensor<ui32>) -> tensor<f32>
    %138 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %139 = stablehlo.multiply %137, %138 : tensor<f32>
    %142 = stablehlo.multiply %104#2, %2 : tensor<f32>
    %143 = stablehlo.divide %142, %2 : tensor<f32>
    %144 = stablehlo.add %76, %143 : tensor<f32>
    %145 = stablehlo.divide %76, %144 : tensor<f32>
    return %145, %133 : tensor<f32>, tensor<2xui64>
  }
}
